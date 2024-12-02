import numpy as np
import pettingzoo as pz
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv
import pygame
from pygame.locals import QUIT
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from pettingzoo.utils import wrappers
import scipy.optimize as opt
from typing import TypeVar, Callable
import jax.numpy as jnp
import torch.nn.functional as F


from civ import Civilization
from reward import RewardCalculator

# Type aliases for readability
State = TypeVar("State")  # Represents the state type
Action = TypeVar("Action")  # Represents the action type

class ProximalPolicyOptimization:
    def __init__(self, env, actor_policies, critic_policies, lambdaa, theta_inits, n_iters, n_fit_trajectories, n_sample_trajectories):
        """
        Initialize PPO with environment and hyperparameters.

        Args:
            env: The environment for training.
            pi: Policy function that maps parameters to a function defining action probabilities.
            lambdaa: Regularization coefficient.
            theta_init: Initial policy parameters.
            n_iters: Number of training iterations.
            n_fit_trajectories: Number of trajectories for fitting the advantage function.
            n_sample_trajectories: Number of trajectories for optimizing the policy.
        """
        self.env = env
        self.actor_policies = actor_policies
        self.critic_policies = critic_policies 
        self.lambdaa = lambdaa
        self.thetas = theta_inits
        self.n_iters = n_iters
        self.n_fit_trajectories = n_fit_trajectories
        self.n_sample_trajectories = n_sample_trajectories

    def train(self):
        """
        Proximal Policy Optimization (PPO) implementation.

        Args:
            env: The environment in which the agent will act.
            actor_policies: RNN defined in test.py
            critic_policies: RNN defined in test.py
            λ: Regularization coefficient to penalize large changes in the policy.
            theta_inits: Initial parameters of the policy, conists of k1 through k10 and epsilon (environmental impact)
            n_iters: Number of training iterations.
            n_fit_trajectories: Number of trajectories to collect for fitting the advantage function.
            n_sample_trajectories: Number of trajectories to collect for optimizing the policy.
        
        Returns:
            Trained policy parameters, theta.
        """

        #This code is taken from class
        theta_list = self.theta_inits
        for _ in range(self.n_iters):

            #used to estimate advantage function
            fit_trajectories=[] #across all agents

            for agent_idx in range(len(self.actor_policies)):
                agent_fit_trajectories = ProximalPolicyOptimization.sample_trajectories( #for a single agent
                    self.env,
                    self.actor_policies[agent_idx],
                    self.critic_policies[agent_idx],
                    self.n_fit_trajectories
                )
                fit_trajectories.append(agent_fit_trajectories)

            A_hat_list = [self.fit(agent_fit_trajectories) for agent_fit_trajectories in fit_trajectories]

            #used to optimize policy
            sample_trajectories_list = [] #across all agents
            for agent_idx, theta in enumerate(theta_list):
                agent_sample_trajectories = ProximalPolicyOptimization.sample_trajectories( #for a single agent
                    self.env,
                    self.pi(theta),
                    self.n_sample_trajectories
                )
                sample_trajectories_list.append(agent_sample_trajectories)
            
            for agent_idx, theta in enumerate(theta_list):
                def objective(theta_opt):
                    total_objective = 0
                    for tau in sample_trajectories_list[agent_idx]:
                        for s, a, _r in tau:
                            pi_curr = self.pi(theta)(s, a)
                            pi_new = self.pi(theta_opt)(s, a)
                            total_objective += pi_new / pi_curr * A_hat_list[agent_idx](s, a) + self.lambdaa * jnp.log(pi_new)
                    return total_objective / self.n_sample_trajectories
                
                theta_list[agent_idx] = self.optimize(objective, theta)

        return theta_list

    def sample_trajectories(env, actor_policies, critic_policies, num_trajectories, max_steps=100):
        """
        Based off of Yu et al.'s 2022 paper on Recurrent MAPPO.
        Collect trajectories by interacting with the environment using recurrent actor and critic networks.

        Args:
            env: The environment to interact with.
            actor_policies: A dictionary mapping each agent to its actor policy function.
            critic_policies: A dictionary mapping each agent to its critic function.
            num_trajectories: Number of trajectories to collect.
            max_steps: Maximum number of steps per trajectory.

        Returns:
            List of trajectories. Each trajectory is a list of tuples containing:
                                                            
            current local observation                 ->  agent's policy ->  probability distribution over actions     ->    action
            actor hidden RNN from previous time step                         actor hidden RNN for current time step

            current state                             ->  value function  -> value estimate for agent
            critic hidden RNN from previous time step                        critic hidden RNN for current time step

            accumulate trajectory into:
            T = (state, obs, actor_hidden_state, critic_hidden_state, action, reward, next_state, next_observation).

            state vs observation: what is observed by all agents vs what is observed by a single agent
            actor hidden state vs critic hidden state: hidden state of an RNN, 
                but one takes into account present state and encoded history in deciding an action
                and the other takes into account present observation and encoded history in deciding a value estimate for 
        """
        trajectories = []

        for _ in range(num_trajectories):
            trajectory = []
            
            # Reset environment and get initial state, converted to usable format
            state = ProximalPolicyOptimization.state_conversion(env.reset())
            
            # Initialize hidden states for actor and critic networks
            actor_hidden_states = {agent: torch.zeros(1, actor_policies[agent].hidden_size) for agent in env.agents}
            critic_hidden_states = {agent: torch.zeros(1, critic_policies[agent].hidden_size) for agent in env.agents}

            for t in range(max_steps):
                actions = {}
                next_actor_hidden_states = {}
                next_critic_hidden_states = {}
                observations = {}
                rewards = {}
                next_states = {}
                dones = {}
                
                for agent in env.agents:
                    # Get agent's observation
                    obs = env.observe(agent)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                    
                    # Actor policy: get action distribution and next hidden state
                    action_probs, next_actor_hidden = actor_policies[agent](obs_tensor, actor_hidden_states[agent])
                    action_dist = Categorical(probs=action_probs)
                    action = action_dist.sample().item()
                    actions[agent] = action
                    next_actor_hidden_states[agent] = next_actor_hidden

                    # Critic policy: get value estimate and next hidden state
                    value, next_critic_hidden = critic_policies[agent](obs_tensor, critic_hidden_states[agent])
                    next_critic_hidden_states[agent] = next_critic_hidden

                # Step environment with all actions
                next_obs, rewards, dones, _ = env.step(actions)

                # Store the current step in the trajectory
                for agent in env.agents:
                    trajectory.append((
                        state, 
                        obs, 
                        actor_hidden_states[agent].detach().cpu().numpy(),
                        critic_hidden_states[agent].detach().cpu().numpy(),
                        actions[agent], 
                        rewards[agent], 
                        next_obs, 
                        next_obs[agent]
                    ))

                # Update hidden states
                actor_hidden_states = next_actor_hidden_states
                critic_hidden_states = next_critic_hidden_states

                # Update state
                state = ProximalPolicyOptimization.state_conversion(next_obs)
                
                # Check if all agents are done
                if all(dones.values()):
                    break

            trajectories.append(trajectory)

        return trajectories

    def state_conversion(state):
        '''
        Converts state to a usable dictionary

        Input: state
        Output: state dictionary indexed by attributes
        '''

        custom_keys = ['ongoing_projects', 'completed_projects', 'explored_tiles', 'captured_cities', 'lost_cities', 'enemy_units_eliminated', 'units_lost', 'GDP', 'energy_output', 'resources_controlled', 'environmental_impact']

        state_dict = {key: None for key in custom_keys}

        state_dict["ongoing_projects"] = 0
        state_dict["completed_projects"]= 0
        state_dict["explored_tiles"]=0
        state_dict['captured_cities']=0
        state_dict['lost_cities']=0
        state_dict['enemy_units_eliminated']=0
        state_dict['units_lost']=0
        state_dict['GDP']=0
        state_dict['energy_output']=0
        state_dict['resources_controlled']=0
        state_dict['environmental_impact']=0

        return state_dict

    def fit(trajectories):
        """
        Fit the advantage function from the given trajectories.

        Args:
            trajectories: A list of trajectories. Each trajectory is a list of (state, action, reward) tuples.

        Returns:
            A_hat: A callable advantage function A_hat(s, a).
        """
        def compute_returns(rewards, gamma=0.99):
            """
            Compute the discounted returns for a trajectory.

            Args:
                rewards: A list of rewards for a single trajectory.
                gamma: Discount factor for future rewards.

            Returns:
                Discounted returns.
            """
            returns = []
            discounted_sum = 0
            for r in reversed(rewards):
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            return returns

        states, actions, rewards = [], [], []
        for trajectory in trajectories:
            for s, a, r in trajectory:
                states.append(s)
                actions.append(a)
                rewards.append(r)

        # Compute returns for each trajectory
        all_returns = []
        for trajectory in trajectories:
            rewards = [r for _, _, r in trajectory]
            all_returns.extend(compute_returns(rewards))

        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(all_returns)

        # Estimate the value function V(s) as the average return for each state
        unique_states = np.unique(states, axis=0)
        state_to_value = {tuple(s): returns[states == s].mean() for s in unique_states}
        V = lambda s: state_to_value.get(tuple(s), 0)

        # Define the advantage function A(s, a) = Q(s, a) - V(s)
        def A_hat(s, a):
            return returns[(states == s) & (actions == a)].mean() - V(s)

        return A_hat

    def optimize(objective, theta_init, method="L-BFGS-B", options=None):
        """
        Optimize the policy parameters to maximize the surrogate objective.

        Args:
            objective: A callable objective function to be maximized.
            theta_init: Initial guess for the policy parameters.
            method: Optimization method (default: L-BFGS-B).
            options: Additional options for the optimizer.

        Returns:
            Optimized policy parameters.
        """
        # Wrapper to negate the objective for minimization (since scipy.optimize minimizes by default)
        def neg_objective(theta):
            return -objective(theta)

        # Perform optimization
        result = opt.minimize(
            neg_objective,
            theta_init,
            method=method,
            options=options or {"disp": True}
        )

        # Return the optimized parameters
        return result.x
    

class ActorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Actor RNN that outputs action probabilities.

        Args:
            input_size: Size of the input observation.
            hidden_size: Size of the RNN hidden state.
            output_size: Number of actions (action space size).
        """
        super(ActorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, observation, hidden_state):
        """
        Forward pass of the actor network.

        Args:
            observation: Input observation tensor of shape (batch_size, input_size).
            hidden_state: Hidden state of the RNN of shape (1, batch_size, hidden_size).

        Returns:
            action_probs: Action probabilities of shape (batch_size, output_size).
            hidden_state: Updated hidden state of the RNN of shape (1, batch_size, hidden_size).
        """
        output, hidden_state = self.rnn(observation.unsqueeze(1), hidden_state)  # Add sequence dimension
        logits = self.fc(output.squeeze(1))  # Remove sequence dimension
        action_probs = F.softmax(logits, dim=-1)  # Convert to probabilities
        return action_probs, hidden_state
    
class CriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Critic RNN that estimates the value of a state.

        Args:
            input_size: Size of the input observation.
            hidden_size: Size of the RNN hidden state.
        """
        super(CriticRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output a single value
    
    def forward(self, observation, hidden_state):
        """
        Forward pass of the critic network.

        Args:
            observation: Input observation tensor of shape (batch_size, input_size).
            hidden_state: Hidden state of the RNN of shape (1, batch_size, hidden_size).

        Returns:
            value: Estimated value of shape (batch_size, 1).
            hidden_state: Updated hidden state of the RNN of shape (1, batch_size, hidden_size).
        """
        output, hidden_state = self.rnn(observation.unsqueeze(1), hidden_state)  # Add sequence dimension
        value = self.fc(output.squeeze(1))  # Remove sequence dimension
        return value, hidden_state
