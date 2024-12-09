import matplotlib.pyplot as plt
import numpy as np
import sys
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
from tqdm import tqdm


from civ import Civilization

# Type aliases for readability
State = TypeVar("State")  # Represents the state type
Action = TypeVar("Action")  # Represents the action type

class ProximalPolicyOptimization:
    def __init__(self, env, actor_policies, critic_policies, lambdaa, n_iters, n_fit_trajectories, n_sample_trajectories, max_steps):
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
            max_steps: Number of iterations to run trajectory for
        """
        self.env = env
        self.actor_policies = actor_policies
        self.critic_policies = critic_policies 
        self.lambdaa = lambdaa
        self.n_iters = n_iters
        self.n_fit_trajectories = n_fit_trajectories
        self.n_sample_trajectories = n_sample_trajectories
        self.max_steps = max_steps


    def train(self, eval_interval, eval_steps):
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
        actor_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.actor_policies.items()
        }

        critic_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.critic_policies.items()
        }
        cumulative_rewards = np.zeros((len(self.env.agents), self.n_iters)) # List to store cumulative rewards per iteration


        for iter in range(self.n_iters):
            print(f"Training iteration: {iter}")
            sys.stdout.flush()

            self.env.reset()   
            
            trajectories = self.initialize_starting_trajectories(self.env, self.actor_policies, len(self.env.agents))

            step=0
            while step<self.max_steps:
                sys.stdout.flush()
                for agent in self.env.agent_iter():
                    sys.stdout.flush()
                    trajectories_next_step = self.sample_trajectories( #for a single agent
                        self.env,
                        self.actor_policies[agent],
                        self.critic_policies[agent],
                        trajectories[agent],
                        iter,
                        agent
                    )
                    cumulative_rewards[agent, iter]+=trajectories_next_step[-3]
                    step+=1
                    if step >= self.max_steps*len(self.env.agents):
                        break
                trajectories[agent].extend(trajectories_next_step)
                
            # Process trajectories
            states = torch.stack([t[0] for t in trajectories])  # States
            obs = torch.stack([self.flatten_observation(t[1]) for t in trajectories])
            actions = torch.stack([t[4] for t in trajectories])  # Actions
            rewards = torch.tensor([t[5] for t in trajectories], dtype=torch.float32)  # Rewards
            returns = self.compute_returns(rewards)  # Compute discounted rewards-to-go

            actor_losses, critic_losses=[],[]

            for agent in self.env.agents:
                # Compute values using critic
                critic_policy = self.critic_policies[agent]
                agent_states = states[agent]
                values, _ = critic_policy(agent_states, None)

                # Compute advantages
                advantages = returns - values.detach()

                # Actor update
                actor_policy = self.actor_policies[agent]
                action_probs, _ = actor_policy(ActorRNN.process_observation(obs[agent]), None)

                # Compute log probabilities and sample actions
                total_log_prob = 0  # To accumulate log probabilities for the entire action
                for key, probs in action_probs.items():
                    dist = Categorical(probs=probs)  # Create a categorical distribution for this component
                    sampled_action = dist.sample()  # Sample an action
                    log_prob = dist.log_prob(sampled_action)  # Get log probability of the sampled action
                    total_log_prob += log_prob  # Accumulate log probabilities

                # Clipped PPO Objective
                ratios = torch.exp(total_log_prob - total_log_prob.detach())
                clipped_ratios = torch.clamp(ratios, 1 - 0.2, 1 + 0.2)
                actor_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

                actor_optimizers[agent].zero_grad()
                actor_loss.backward()
                actor_optimizers[agent].step()

                actor_losses.append(actor_loss.item())

                # Critic update
                critic_loss = F.mse_loss(values, returns)
                critic_optimizers[agent].zero_grad()
                critic_loss.backward()
                critic_optimizers[agent].step()

                critic_losses.append(critic_loss.item())

            # Evaluation
            if (iter + 1) % eval_interval == 0:
                self.env.reset()
                step = 0
                
                for agent in self.env.agent_iter():
                    print("Step", step)
                    sys.stdout.flush()
                    # Observe and select action
                    observation = self.env.observe(agent)
                    obs_tensor = ActorRNN.process_observation(observation)
                    with torch.no_grad():
                        action_probs, _ = self.actor_policies[agent](obs_tensor, None)
                        chosen_action = self.sample_action(action_probs)
                    self.env.step(chosen_action)
                    self.env.render()

                    if agent == self.env.agents[-1]:  # Check if this is the last agent for the step
                        step += 1
                    if step >= eval_steps:
                        break



        plt.figure(figsize=(10, 6))
        for agent in range(len(self.env.agents)):
            plt.plot(range(self.n_iters), cumulative_rewards[agent, :], label=f"Agent {agent + 1}")
        plt.xlabel("Training Iterations")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Over Training Iterations")
        plt.legend()
        plt.grid()
        plt.show()
        return actor_losses[-1], critic_losses[-1]
    
    def sample_trajectories(self,env,actor_policy,critic_policy,past_trajectory,iter,agent):
        """
        Based off of Yu et al.'s 2022 paper on Recurrent MAPPO.
        Collect trajectories by interacting with the environment using recurrent actor and critic networks.

        Args:
            env: The environment to interact with.
            actor_policies: A dictionary mapping each agent to its actor policy function.
            critic_policies: A dictionary mapping each agent to its critic function.
            past_trajectory: The past trajectory for this agent

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
               

        trajectories_next_step=[] #ends up being a list of num_agents length, where each element is what gets added to existing trajectory     
        
        #past_trajectory's last elements include state, obs, hidden actor, hidden critic, action, reward next time step, state next time step, obs next time step
        actor_hidden_state = past_trajectory[-6] 
        critic_hidden_state = past_trajectory[-5]
        state_t = past_trajectory[-2]
        obs_t = past_trajectory[-1]


        obs_t_tensor = ActorRNN.process_observation(obs_t)

        # Actor policy: get action distribution and next hidden state
        action_probs, next_actor_hidden = actor_policy(obs_t_tensor, actor_hidden_state)
        action = self.sample_action(action_probs)

        # Step environment with all actions
        agent = env.agent_selection
        env.step(action)  # Pass a dictionary with the agent and its action
        reward_t = env.rewards[agent]

        done = env.dones[agent]
        next_obs = env.observe(agent)
        next_state = self.get_global_state(env)

        # Critic policy: get value and next critic hidden state
        value, next_critic_hidden = critic_policy(state_t, critic_hidden_state)

        trajectories_next_step=[
            state_t,
            obs_t, 
            next_actor_hidden,
            next_critic_hidden, 
            action,
            reward_t, 
            next_state,
            next_obs
        ]

        return trajectories_next_step
    
    def get_global_state(self,env):

        obs_for_critic = []

        # Gather observations for all agents
        for agent_idx in env.agents:
            agent_obs = env.observe(env.agents[agent_idx])  # Get the local observation for the agent
            obs_for_critic.append(agent_obs)  

        # Critic policy: get value estimate and next hidden state
        critic_dict = {}
        # this is stupid naming, this is just the mask
        critic_visibility = env.get_full_masked_map()
        map_copy = env.map.copy()
        # do the masking
        critic_map = np.where(critic_visibility[:, :, np.newaxis].squeeze(2), map_copy, np.zeros_like(map_copy))
        #print(critic_map.shape)
        # THIS MIGHT FUCK THINGS UP IN THE FUTURE. 
        critic_dict['map'] = critic_map
        critic_dict['units'] = None
        critic_dict['cities'] = None
        critic_dict['money'] = None
        for obs in obs_for_critic:
            for key in obs:
                if key != 'map':
                    if critic_dict[key] is None: 
                        critic_dict[key] = obs[key]
                    else:
                        critic_dict[key] = np.concatenate((critic_dict[key], obs[key]), axis=0) #idk if you can concatenate spaces.box
                else: 
                    pass

        for key in critic_dict: 
            critic_dict[key] = torch.tensor(critic_dict[key], dtype=torch.float32).flatten()
        state = torch.cat([critic_dict[key] for key in critic_dict]).unsqueeze(0)    
        return state

    def initialize_starting_trajectories(self,env, actor_policies, n_agents):
        """
        Initialize the starting trajectories for each agent.

        Args:
            env: The environment.
            actor_policies: A dictionary of actor policies for each agent.
            critic_policies: A dictionary of critic policies for each agent.

        Returns:
            A list of initial trajectories, one for each agent.
        """
        starting_trajectories = []

        n_agents = len(env.agents)
        initial_state = self.get_global_state(env) # Global state

        for agent_idx, agent in enumerate(env.agents):
            # Initial state and observation for each agent
            initial_observation = env.observe(agent)  # Local observation for the agent

            # Initialize hidden states for actor and critic networks
            input_size = actor_policies[agent_idx].hidden_size
            actor_hidden_state = torch.zeros(1, 1, input_size)  # Shape: (1, batch_size, hidden_size)
            critic_hidden_state = torch.zeros(1, 1, input_size)

            # Placeholder for the rest of the trajectory components
            initial_action = torch.zeros((5,), dtype=torch.float32)  # Adjust `action_space_size` as needed
            initial_reward = 0  # No reward has been received yet
            initial_next_state = initial_state  # Initial state is also the "next state"
            initial_next_observation = initial_observation  # Same for the observation

            # Create the initial trajectory structure
            starting_trajectory = [
                initial_state,             # State at time t
                initial_observation,       # Observation at time t
                actor_hidden_state,        # Actor hidden state
                critic_hidden_state,       # Critic hidden state
                initial_action,            # Action at time t (None at start)
                initial_reward,            # Reward at time t (0 at start)
                initial_next_state,        # Next state (same as initial state)
                initial_next_observation   # Next observation (same as initial observation)
            ]
            starting_trajectories.append(starting_trajectory)

        return starting_trajectories


    def sample_action(self, action_probs):
        # Sample action components
        action_type_dist = Categorical(probs=action_probs['action_type'])
        action_type = action_type_dist.sample().item()

        unit_id_dist = Categorical(probs=action_probs['unit_id'])
        unit_id = unit_id_dist.sample().item()

        direction_dist = Categorical(probs=action_probs['direction'])
        direction = direction_dist.sample().item()

        city_id_dist = Categorical(probs=action_probs['city_id'])
        city_id = city_id_dist.sample().item()

        project_id_dist = Categorical(probs=action_probs['project_id'])
        project_id = project_id_dist.sample().item()

        action = {
            'action_type': action_type,
            'unit_id': unit_id,
            'direction': direction,
            'city_id': city_id,
            'project_id': project_id,
        }
        return action
    def flatten_observation(self, observation):
        """
        Flattens a dictionary of observations into a single tensor.

        Args:
            observation (dict): A dictionary of observations.

        Returns:
            torch.Tensor: A flattened tensor of concatenated observation values.
        """
        tensors = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):  # Convert NumPy arrays to tensors
                value = torch.tensor(value, dtype=torch.float32)
            elif not isinstance(value, torch.Tensor):  # Skip non-tensor types
                continue
            tensors.append(value.flatten())  # Flatten each tensor
        return torch.cat(tensors)  # Concatenate into a single tensor

    def compute_returns(self, rewards, gamma=0.99):
            """
            Compute the discounted rewards-to-go (returns) for a given list of rewards.

            Args:
                rewards (torch.Tensor): A 1D tensor of rewards for a trajectory.
                gamma (float): Discount factor for future rewards. Default is 0.99.

            Returns:
                torch.Tensor: A 1D tensor of discounted rewards-to-go (returns).
            """
            returns = torch.zeros_like(rewards)
            discounted_sum = 0.0

            # Calculate the returns in reverse order
            for t in reversed(range(len(rewards))):
                discounted_sum = rewards[t] + gamma * discounted_sum
                returns[t] = discounted_sum

            return returns
    

class ActorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, max_units_per_agent, max_cities, max_projects):
        super(ActorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_action_type = nn.Linear(hidden_size, 7)
        self.fc_unit_id = nn.Linear(hidden_size, max_units_per_agent)
        self.fc_direction = nn.Linear(hidden_size, 4)
        self.fc_city_id = nn.Linear(hidden_size, max_cities)
        self.fc_project_id = nn.Linear(hidden_size, max_projects)
            
    def process_observation(obs):
        if isinstance(obs, dict):
            processed_obs = []
            for key in obs:
                value = obs[key]
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    tensor_value = torch.tensor(value, dtype=torch.float32).flatten()
                    processed_obs.append(tensor_value)
                else:
                    # Handle other types if necessary
                    pass
            obs_tensor = torch.cat(processed_obs).unsqueeze(0)
        elif isinstance(obs, np.ndarray) or isinstance(obs, torch.Tensor):
            # If obs is already a tensor or array, flatten and unsqueeze
            obs_tensor = torch.tensor(obs, dtype=torch.float32).flatten().unsqueeze(0)
        else:
            # Handle other types if necessary
            raise TypeError(f"Unsupported observation type: {type(obs)}")
        return obs_tensor


    def forward(self, observation, hidden_state):
        output, hidden_state = self.rnn(observation.unsqueeze(1), hidden_state)  # Add sequence dimension
        output = output.squeeze(1)  # Remove sequence dimension

        action_type_logits = self.fc_action_type(output)
        action_type_probs = F.softmax(action_type_logits, dim=-1)

        unit_id_logits = self.fc_unit_id(output)
        unit_id_probs = F.softmax(unit_id_logits, dim=-1)

        direction_logits = self.fc_direction(output)
        direction_probs = F.softmax(direction_logits, dim=-1)

        city_id_logits = self.fc_city_id(output)
        city_id_probs = F.softmax(city_id_logits, dim=-1)

        project_id_logits = self.fc_project_id(output)
        project_id_probs = F.softmax(project_id_logits, dim=-1)

        action_probs = {
            'action_type': action_type_probs,
            'unit_id': unit_id_probs,
            'direction': direction_probs,
            'city_id': city_id_probs,
            'project_id': project_id_probs,
        }

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