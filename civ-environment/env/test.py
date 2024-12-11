import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from civ import Civilization
from pettingzoo.utils import wrappers
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Import the PPO class and the Actor/Critic RNNs from your implementation
from train import ProximalPolicyOptimization
from rnn import ActorRNN, CriticRNN

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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Define hyperparameters
    hidden_size = 1024
    lambdaa = 0.01
    step_max = 100 #number of training iterations, has to be over 10 in order to save outputs
    n_fit_trajectories = 100 #have to delete
    n_sample_trajectories = 100 #have to delete
    T = 500 #number of steps in trajectory
    num_epochs = 100
    eval_interval=step_max 
    eval_steps=T #number of steps to run civ game for
    batch_size = 5  # Define your batch size
    K = 10 #number of minibatches to process

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
        actor_policies[agent] = ActorRNN(input_size, hidden_size, action_size, env.max_cities, env.max_projects, device)
    critic_policies = CriticRNN(input_size_critic, hidden_size, device)

    # Instantiate PPO
    ppo = ProximalPolicyOptimization(
        env=env,
        actor_policies=actor_policies,
        critic_policies=critic_policies,
        lambdaa=lambdaa,
        step_max=step_max,
        n_fit_trajectories=n_fit_trajectories,
        n_sample_trajectories=n_sample_trajectories,
        T = T, 
        batch_size = batch_size,
        K = K,
        device=device
    )

    # Train the policies
    ppo.train(eval_interval, eval_steps)

if __name__ == "__main__":
    main()