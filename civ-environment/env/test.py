import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from civ import Civilization
from pettingzoo.utils import wrappers
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Import the PPO class and the Actor/Critic RNNs from your implementation
from train import ProximalPolicyOptimization, ActorRNN, CriticRNN

def preprocess_observation(obs):
    """
    Flatten the observation dictionary into a tensor.
    """
    # Flatten and concatenate all components of the observation
    map_obs = torch.tensor(obs['map'], dtype=torch.float32).flatten()
    units_obs = torch.tensor(obs['units'], dtype=torch.float32).flatten()
    cities_obs = torch.tensor(obs['cities'], dtype=torch.float32).flatten()
    money_obs = torch.tensor(obs['money'], dtype=torch.float32).flatten()
    obs_tensor = torch.cat([map_obs, units_obs, cities_obs, money_obs])
    return obs_tensor

def main():
    # Initialize the environment
    map_size = (15, 30)
    num_agents = 4
    env = Civilization(map_size, num_agents)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

    # Define hyperparameters
    hidden_size = 128
    lambdaa = 0.01
    n_iters = 10
    n_fit_trajectories = 5
    n_sample_trajectories = 5

    # Initialize policies and optimizers
    actor_policies = {}
    critic_policies = {}
    theta_inits = {}
    env.reset()
    for agent in env.agents:
        # Get observation and action spaces
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)

        # Calculate input size for networks
        input_size = np.prod(obs_space['map'].shape) + \
                     np.prod(obs_space['units'].shape) + \
                     np.prod(obs_space['cities'].shape) + \
                     np.prod(obs_space['money'].shape)
        input_size_critic = np.prod(obs_space['map'].shape) + \
                     num_agents * np.prod(obs_space['units'].shape) + \
                     num_agents * np.prod(obs_space['cities'].shape) + \
                     num_agents * np.prod(obs_space['money'].shape)

        # Calculate output size for the actor (number of discrete actions)
        action_size = act_space['action_type'].n

        # Initialize actor and critic networks
        actor_policies[agent] = ActorRNN(input_size, hidden_size, action_size, env.max_cities, env.max_projects)
        critic_policies[agent] = CriticRNN(input_size_critic, hidden_size)

        # Initialize policy parameters
        theta_inits[agent] = np.random.randn(sum(p.numel() for p in actor_policies[agent].parameters()))

    # Instantiate PPO
    ppo = ProximalPolicyOptimization(
        env=env,
        actor_policies=actor_policies,
        critic_policies=critic_policies,
        lambdaa=lambdaa,
        theta_inits=theta_inits,
        n_iters=n_iters,
        n_fit_trajectories=n_fit_trajectories,
        n_sample_trajectories=n_sample_trajectories
    )

    # Train the policies
    ppo.train()

if __name__ == "__main__":
    main()