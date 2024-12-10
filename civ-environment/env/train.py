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
import random
from rnn import ActorRNN, CriticRNN


from civ import Civilization

# Type aliases for readability
State = TypeVar("State")  # Represents the state type
Action = TypeVar("Action")  # Represents the action type

class ProximalPolicyOptimization:
    def __init__(self, env, actor_policies, critic_policies, lambdaa, step_max, n_fit_trajectories, n_sample_trajectories, T, batch_size, K):
        """
        Initialize PPO with environment and hyperparameters.

        Args:
            env: The environment for training.
            pi: Policy function that maps parameters to a function defining action probabilities.
            lambdaa: Regularization coefficient.
            theta_init: Initial policy parameters.
            step_max: Number of training iterations.
            n_fit_trajectories: Number of trajectories for fitting the advantage function.
            n_sample_trajectories: Number of trajectories for optimizing the policy.
            T: Number of iterations to run trajectory for
        """
        self.env = env
        self.actor_policies = actor_policies
        self.critic_policies = critic_policies 
        self.lambdaa = lambdaa
        self.step_max = step_max
        self.n_fit_trajectories = n_fit_trajectories
        self.n_sample_trajectories = n_sample_trajectories
        self.T = T
        self.batch_size = batch_size
        self.K = K


    def train(self, eval_interval, eval_steps):
        """
        Proximal Policy Optimization (PPO) implementation.

        Args:
            env: The environment in which the agent will act.
            actor_policies: RNN defined in test.py
            critic_policies: RNN defined in test.py
            λ: Regularization coefficient to penalize large changes in the policy.
            theta_inits: Initial parameters of the policy, conists of k1 through k10 and epsilon (environmental impact)
            step_max: Number of training iterations.
            n_fit_trajectories: Number of trajectories to collect for fitting the advantage function.
            n_sample_trajectories: Number of trajectories to collect for optimizing the policy.
        
        Returns:
            Trained policy parameters, theta.
        """
        
        #each agent in the environment is assigned a separate optimizer for both their actor policy and critic policy networks.
        actor_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.actor_policies.items()
        }

        critic_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.critic_policies.items()
        }

        cumulative_rewards = np.zeros((len(self.env.agents), self.step_max)) # List to store cumulative rewards per iteration

        for step in range(self.step_max):
            print(f"Training iteration: {step}")
            sys.stdout.flush()

            D = []

            for i in range(self.batch_size):

                self.env.reset()   
                
                trajectories, cumulative_rewards = self.generate_all_trajectories(cumulative_rewards, step)
                    
                D.append(trajectories)

                old_action_probs = self.compute_old_action_probs(trajectories)
                old_action_probs = [{k: v.detach() for k,v in a.items()} for a in old_action_probs]

                # Compute advantages
                A_hat = self.fit(trajectories)

                D.append(trajectories)

            for minibatch in range(int(self.K)): #for now, just a single trajectory
                random_mini_batch = random.choice(D) #TO CHANGE: mini batch is 1 rn

                chunk_size = 45 # TO CHANGE: just doing every 5 time steps

                data_chunks = [random_mini_batch[i:i+chunk_size] for i in range(0, len(random_mini_batch), chunk_size)]

                actor_hiddens = [step[2] for step in random_mini_batch]
                critic_hiddens = [step[3] for step in random_mini_batch]

                for data_chunk in data_chunks:
                    # update RNN hidden states for π and V from first hidden state in data chunk and propagate
                    probs, actor_hiddens, values,critic_hiddens= self.updateRNN(data_chunk, actor_hiddens, critic_hiddens)

            # ACTOR LOSS UPDATES
            self.actor_adam(trajectories, actor_optimizers, old_action_probs, probs, A_hat)
            old_action_probs = probs
            self.critic_adam(trajectories, critic_optimizers, critic_hiddens)

            self.evaluate(step, eval_interval, eval_steps)

        plt.figure(figsize=(10, 6))
        for agent in range(len(self.env.agents)):
            plt.plot(range(self.step_max), cumulative_rewards[agent, :], label=f"Agent {agent + 1}")
        plt.xlabel("Training Iterations")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Over Training Iterations")
        plt.legend()
        plt.grid()
        plt.show()
        return None
    
    def generate_all_trajectories(self, cumulative_rewards, step):
        trajectories = self.initialize_starting_trajectories(self.env, self.actor_policies)

        t=0
        while t<self.T:
            sys.stdout.flush()
            for agent in self.env.agent_iter():
                sys.stdout.flush()
                trajectories_next_step = self.generate_single_trajectory( #for a single agent
                    self.env,
                    self.actor_policies[agent],
                    self.critic_policies[agent],
                    trajectories[agent],
                    agent
                )
                cumulative_rewards[agent, step]+=trajectories_next_step[-4]
                trajectories[agent].extend(trajectories_next_step)
                t+=1
                if t >= self.T*len(self.env.agents):
                    break
            #shape: # agents x length of trajectory
        
        return trajectories,cumulative_rewards
    
    def generate_single_trajectory(self,env,actor_policy,critic_policy,past_trajectory,agent):
            
        trajectories_next_step=[] #ends up being a list of num_agents length, where each element is what gets added to existing trajectory     
        
        #past_trajectory's last elements include state, obs, hidden actor, hidden critic, action, reward next time step, state next time step, obs next time step
        actor_hidden_state = past_trajectory[-7] 
        critic_hidden_state = past_trajectory[-6]
        state_t = past_trajectory[-3]
        obs_t_dict = past_trajectory[-2]
        obs_t = self.flatten_observation(obs_t_dict)  # convert dict to tensor
        obs_t_input = obs_t.unsqueeze(0).unsqueeze(0)

        # Prepare inputs for RNNs: (batch_size=1, seq_len=1, input_size)
        obs_t_input = obs_t.unsqueeze(0).unsqueeze(0)  # (1,1,input_size)
        state_t_input = state_t.unsqueeze(0).unsqueeze(0)  # (1,1,input_size_critic)


        # Actor policy: get action distribution and next hidden state
        action_probs, next_actor_hidden = actor_policy(obs_t_input, actor_hidden_state)
        action = self.sample_action(action_probs)

        # Step environment with all actions
        agent = env.agent_selection
        env.step(action)  # Pass a dictionary with the agent and its action
        reward_t = env.rewards[agent]

        done = env.dones[agent]
        next_obs = env.observe(agent)
        next_state = self.get_global_state(env)

        # Critic policy: get value and next critic hidden state
        value, next_critic_hidden = critic_policy(state_t_input, critic_hidden_state)

        trajectories_next_step=[
            state_t,
            obs_t, 
            next_actor_hidden,
            next_critic_hidden, 
            action,
            reward_t, 
            next_state,
            next_obs,
            value
        ]

        return trajectories_next_step
    
    def updateRNN(self,trajectories, initial_actor_hidden, initial_critic_hidden):
        states = torch.stack([
            t
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 0  # Keep only the 3rd element (index 2 of each 9-element chunk)
        ])
        obs = torch.stack([
            self.flatten_observation(t)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 1  # Keep only the observation at index 1 of each 9-element chunk
        ])
        
        across_agent_probs, across_agent_final_actor_hiddens, across_agent_values, across_agent_final_critic_hiddens=[],[],[],[]

        #update RNN hidden states for π and V from first hidden state in data chunk
        for agent in self.env.agents:

            agent_states = states[agent]  
            agent_obs = obs[agent]

            actor_input = agent_obs.unsqueeze(0).unsqueeze(0)
            critic_input = agent_states.unsqueeze(0).unsqueeze(0)



            probs, final_actor_hidden = self.actor_policies[agent](actor_input, initial_actor_hidden[agent])
            values, final_critic_hidden = self.critic_policies[agent](critic_input, initial_critic_hidden[agent])
            across_agent_probs.append(probs)
            across_agent_final_actor_hiddens.append(final_actor_hidden)
            across_agent_values.append(values)
            across_agent_final_critic_hiddens.append(final_critic_hidden)

        return across_agent_probs, across_agent_final_actor_hiddens, across_agent_values, across_agent_final_critic_hiddens
    
    def fit(self, trajectories,gamma = 0.99, lam = 0.95):
        # Process trajectories
        rewards = [[t for i, t in enumerate(agent_trajectories) if (i % 9) == 5]
                   for agent_trajectories in trajectories]

        # Convert to a 2D tensor
        rewards = torch.tensor(rewards, dtype=torch.float32)
        critic_hiddens = torch.stack([
            t
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 3  # Keep only the 3rd element (index 2 of each 9-element chunk)
        ])
        states = torch.stack([
            t
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 0  # Keep only the 3rd element (index 2 of each 9-element chunk)
        ])

        # Compute discounted returns-to-go
        returns = self.compute_returns(rewards, gamma)  # Shape: (n_agents * T,)

        # Initialize tensor to store critic values
        values = torch.zeros_like(returns)

        for agent in self.env.agents:
            agent_values = []

            # Compute critic values (V(s)) for each state (index 0 in trajectory step)
            with torch.no_grad():
                agent_states = states[agent]
                agent_hiddens = critic_hiddens[agent]
                
                state_input = agent_states.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, state_dim)
                val, _ = self.critic_policies[agent](state_input, agent_hiddens)
                # val shape: (1, 1), take val.squeeze()
                agent_values.append(val.squeeze())

            agent_values_tensor = torch.stack(agent_values, dim=0)  # shape: (T,)

        # Compute returns using the rewards and the last value of zero (or bootstrap if you have next_value)
        # For GAE, we need values and next values. We'll treat next_value at T as 0 for simplicity.
        # We'll compute advantages per agent:
        advantages = torch.zeros_like(values)

        for agent in self.env.agents:
            agent_rewards = rewards[agent]
            agent_values = values[agent]

            # Generalized Advantage Estimation (GAE)
            # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            # advantage_t = delta_t + gamma * lam * advantage_{t+1}
            gae = 0.0
            for t in reversed(range(self.T)):
                if t == self.T - 1:
                    next_value = 0.0
                else:
                    next_value = agent_values[t + 1].item()

                delta = agent_rewards[t].item() + gamma * next_value - agent_values[t].item()
                gae = delta + gamma * lam * gae
                advantages[agent, t] = gae

        # Normalize advantages across all agents and timesteps if desired
        flat_adv = advantages.flatten()
        advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        return advantages
    def compute_log_prob(self, distributions, action):
        # 'distributions' might be a dict or tuple containing separate distributions for each action dimension.
        # 'action' could be a dict or tuple with the chosen actions for each dimension.

        # Example assuming 'distributions' is a dict of Torch distributions and
        # 'action' is a dictionary with keys matching the distributions.
        log_prob = 0.0
        log_prob += distributions['action_type'].log_prob(torch.tensor(action['action_type']))
        log_prob += distributions['unit_id'].log_prob(torch.tensor(action['unit_id']))
        log_prob += distributions['direction'].log_prob(torch.tensor(action['direction']))
        log_prob += distributions['city_id'].log_prob(torch.tensor(action['city_id']))
        log_prob += distributions['project_id'].log_prob(torch.tensor(action['project_id']))
        return log_prob

    def actor_adam(self, trajectories, actor_optimizers, old_action_probs, new_action_probs, A_hat):
        obs = torch.stack([
            self.flatten_observation(t)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 1  # Keep only the observation at index 1 of each 9-element chunk
        ])

   
        actions = []
        for agent_trajectories in trajectories:
            # Extract actions (every 9th element starting at index 4)
            agent_actions = [t for i, t in enumerate(agent_trajectories) if i % 9 == 4]
            actions.append(agent_actions)
        

        for agent, agent_name in enumerate(self.env.agents):
            # Extract the per-agent trajectories, advantages, obs, actions
            agent_trajectory = trajectories[agent]
            agent_advantages = A_hat[agent]
            agent_obs = obs[agent]
            agent_actions = actions[agent]  # actions should be a tensor or dict per timestep

            new_log_probs = []
            old_log_probs = []

            # For every time step
            for t in range(self.T):
                # Retrieve the agent's chosen action at time t
                # If you've stored them as dictionaries:
                chosen_action = {
                    'action_type': agent_actions[t]['action_type'],
                    'unit_id': agent_actions[t]['unit_id'],
                    'direction': agent_actions[t]['direction'],
                    'city_id': agent_actions[t]['city_id'],
                    'project_id': agent_actions[t]['project_id']
                }
                # Compute the log probability from the new and old policies
                new_lp = self.compute_log_prob(new_action_probs[agent], chosen_action)
                old_lp = self.compute_log_prob(old_action_probs[agent], chosen_action)

                new_log_probs.append(new_lp)
                old_log_probs.append(old_lp)

            # Convert lists to tensors
            new_log_probs = torch.stack(new_log_probs)
            old_log_probs = torch.stack(old_log_probs)

            # Compute ratio and PPO loss as usual
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
            actor_loss = -torch.min(ratio * agent_advantages, clipped_ratio * agent_advantages).mean()

            actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            actor_optimizers[agent].step()

    def critic_adam(self, trajectories, critic_optimizers, critic_hidden, gamma=0.99, epsilon=0.1):
        """
        Perform critic updates using the PPO clipped value loss.

        Args:
            trajectories: List of trajectories for all agents.
            critic_optimizers: Dictionary of optimizers for each agent's critic network.
            critic_hidden: Dictionary of critic hidden states for each agent.
            gamma: Discount factor for returns.
            epsilon: Clipping parameter for value loss.
        """
        # Extract states, rewards, and compute returns
        states = torch.tensor([
            t
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 0  # Keep only the 3rd element (index 2 of each 9-element chunk)
        ])
        rewards = torch.tensor([
            t
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 5  # Keep only the 3rd element (index 2 of each 9-element chunk)
        ])
        returns = self.compute_returns(rewards, gamma)  # Discounted returns-to-go

        for agent in self.env.agents:
            # Extract agent-specific data
            agent_states = states[agent]  # States for the agent
            agent_returns = returns[agent]  # Returns for the agent
            agent_critic_hidden = critic_hidden[agent]  # Hidden state for the agent

            # Forward pass through the critic network
            values, _ = self.critic_policies[agent](agent_states, agent_critic_hidden)

            # Compute unclipped and clipped value losses
            value_loss_unclipped = (values.squeeze(1) - agent_returns) ** 2
            clipped_values = torch.clamp(
                values.squeeze(1),
                min=(agent_returns - epsilon),
                max=(agent_returns + epsilon),
            )
            value_loss_clipped = (clipped_values - agent_returns) ** 2

            # Compute critic loss by taking the maximum loss
            agent_critic_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # Backpropagation and optimizer step
            critic_optimizers[agent].zero_grad()
            agent_critic_loss.backward()
            critic_optimizers[agent].step()

    def evaluate(self, step, eval_interval, eval_steps):
        # Evaluation
        if (step + 1) % eval_interval == 0:
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

    def compute_old_action_probs(self, trajectories):
        """
        Creates a copy of the actor policies for all agents, with the same structure and parameters.

        Returns:
            dict: A dictionary containing the copied actor policies for each agent.
        """
        obs = torch.stack([
            self.flatten_observation(t)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 1  # Keep only the observation at index 1 of each 9-element chunk
        ])
        actor_hidden_states = torch.stack([
            t if isinstance(t, torch.Tensor) else torch.tensor(t)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 2  # Keep only the actor hidden states
        ])
                    

        old_actor_policies = {
            agent: type(self.actor_policies[agent])(
                self.actor_policies[agent].rnn.input_size,
                self.actor_policies[agent].hidden_size,
                self.actor_policies[agent].fc_unit_id.out_features,
                self.actor_policies[agent].fc_city_id.out_features,
                self.actor_policies[agent].fc_project_id.out_features
            ) for agent in self.env.agents
        }

        # Copy the parameters
        for agent in self.env.agents:
            old_actor_policies[agent].load_state_dict(self.actor_policies[agent].state_dict())
            
        across_agents_actions_probs =[]
        for agent in self.env.agents:   
            obs_agent = obs[agent].unsqueeze(0).unsqueeze(0)  

            action_probs, _ = old_actor_policies[agent](obs_agent, actor_hidden_states[agent])  # Actor hidden state
            across_agents_actions_probs.append(action_probs)

        return across_agents_actions_probs

    def compute_log_prob(self, action_probs, action):

        print("Action probabilities:", action_probs)
        print("Chosen action:", action)

        total_log_prob = 0.0
        for key, probs in action_probs.items():
            dist = Categorical(probs=probs)
            selected_action = torch.tensor(action[key], dtype=torch.long)
            log_prob = dist.log_prob(selected_action)
            total_log_prob += log_prob
        return total_log_prob
        
    def get_global_state(self, env):
        # Gather all agent observations to construct a global state
        obs_for_critic = []
        for agent in env.agents:
            agent_obs = env.observe(agent)
            obs_for_critic.append(agent_obs)

        critic_dict = {}
        critic_visibility = env.get_full_masked_map()
        map_copy = env.map.copy()
        critic_map = np.where(critic_visibility[:, :, np.newaxis].squeeze(2), map_copy, np.zeros_like(map_copy))
        critic_dict['map'] = torch.tensor(critic_map, dtype=torch.float32).flatten()

        # Concatenate units, cities, money across agents
        units_list = []
        cities_list = []
        money_list = []
        for obs in obs_for_critic:
            units_list.append(torch.tensor(obs['units'], dtype=torch.float32).flatten())
            cities_list.append(torch.tensor(obs['cities'], dtype=torch.float32).flatten())
            money_list.append(torch.tensor(obs['money'], dtype=torch.float32).flatten())

        critic_dict['units'] = torch.cat(units_list)
        critic_dict['cities'] = torch.cat(cities_list)
        critic_dict['money'] = torch.cat(money_list)

        state = torch.cat([critic_dict[k] for k in critic_dict]).float()
        return state

    def initialize_starting_trajectories(self,env, actor_policies):
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
            initial_observation = ActorRNN.process_observation(initial_observation)

            # Initialize hidden states for actor and critic networks
            input_size = actor_policies[agent_idx].hidden_size
            actor_hidden_state = torch.zeros(1, 1, input_size)  # Shape: (1, batch_size, hidden_size)
            critic_hidden_state = torch.zeros(1, 1, input_size)
        
            action_type = random.choice([env.MOVE_UNIT, env.ATTACK_UNIT, env.FOUND_CITY, env.ASSIGN_PROJECT, env.NO_OP])

            # Placeholder for the rest of the trajectory components
            initial_action = {
                "action_type": action_type,
                "unit_id": random.randint(0, len(env.units[agent]) - 1) if env.units[agent] else 0,
                "direction": random.randint(0, 3),
                "city_id": random.randint(0, len(env.cities[agent]) - 1) if env.cities[agent] else 0,
                "project_id": random.randint(0, env.max_projects - 1)
            }
            initial_reward = 0  # No reward has been received yet
            initial_next_state = initial_state  # Initial state is also the "next state"
            initial_next_observation = initial_observation  # Same for the observation
            initial_value = 0

            # Create the initial trajectory structure
            starting_trajectory = [
                initial_state,             # State at time t
                initial_observation,       # Observation at time t
                actor_hidden_state,        # Actor hidden state
                critic_hidden_state,       # Critic hidden state
                initial_action,            # Action at time t (None at start)
                initial_reward,            # Reward at time t (0 at start)
                initial_next_state,        # Next state (same as initial state)
                initial_next_observation,   # Next observation (same as initial observation)
                initial_value
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
        if isinstance(observation, dict):
            # Flatten dictionary values
            tensors = []
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    value = torch.tensor(value, dtype=torch.float32)
                elif not isinstance(value, torch.Tensor):
                    # handle other cases if necessary
                    pass
                tensors.append(value.flatten())
            return torch.cat(tensors)
        elif isinstance(observation, torch.Tensor):
            # Already a tensor, just return it as is (or flatten if needed)
            return observation.flatten()
        else:
            # Handle other unexpected types if necessary
            raise TypeError(f"Unsupported observation type: {type(observation)}")


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

        for agent in self.env.agents:
            # Calculate the returns in reverse order
            for t in reversed(range(len(rewards[agent]))):
                discounted_sum = rewards[agent,t] + gamma * discounted_sum
                returns[agent, t] = discounted_sum

        return returns
    
