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

from civ import Civilization
from reward import RewardCalculator

# Type aliases for readability
State = TypeVar("State")  # Represents the state type
Action = TypeVar("Action")  # Represents the action type

class ProximalPolicyOptimization:
    def __init__(self, env, pi, lambdaa, theta_init, n_iters, n_fit_trajectories, n_sample_trajectories):
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
        self.pi = pi
        self.lambdaa = lambdaa
        self.theta = theta_init
        self.n_iters = n_iters
        self.n_fit_trajectories = n_fit_trajectories
        self.n_sample_trajectories = n_sample_trajectories

    def train(self):
        """
        Proximal Policy Optimization (PPO) implementation.

        Args:
            env: The environment in which the agent will act.
            pi: Policy function that maps parameters to a function defining action probabilities.
            λ: Regularization coefficient to penalize large changes in the policy.
            theta_init: Initial parameters of the policy, conists of k1 through k10 and epsilon (environmental impact)
            n_iters: Number of training iterations.
            n_fit_trajectories: Number of trajectories to collect for fitting the advantage function.
            n_sample_trajectories: Number of trajectories to collect for optimizing the policy.
        
        Returns:
            Trained policy parameters, theta.
        """
        # Initialize policy parameters
        theta = self.theta_init
        
        # Main training loop
        for k in range(self.n_iters):
            # Collect trajectories for fitting the advantage function
            fit_trajectories = sample_trajectories(self.env, self.pi(theta), self.n_fit_trajectories)
            
            # Fit the advantage function based on collected trajectories
            # A_hat is a function that estimates the advantage of actions
            A_hat = self.fit(fit_trajectories)

            # Collect trajectories for optimizing the policy
            sample_trajectories = sample_trajectories(self.env, self.pi(theta), self.n_sample_trajectories)
            
            # Define the objective function, which is reward we want to maximize
            def objective(theta_opt):
                """
                objective function for PPO.

                Args:
                    theta_opt: The policy parameters being optimized.

                Returns:
                    Total objective value for the given theta_opt.
                """

                total_objective = 0

                for tau in sample_trajectories:  # Loop over sampled trajectories
                    for s, a, s_next in tau:  # Each trajectory consists of (state, action, next state)
                        # Compute reward using the detailed reward function
                        R = RewardCalculator.compute_reward(s, s_next, theta_opt)

                        # Advantage function incorporating the new reward
                        A_hat = R + self.epsilon * self.A_previous(s, a)  # A_previous is the estimated advantage function

                        # Policy ratio for PPO
                        policy_ratio =self.pi(theta_opt)(s, a) / self.pi(theta)(s, a)

                        # Add to the total objective with PPO clipping (optional)
                        total_objective += policy_ratio * A_hat + self.lambdaa * jnp.log(self.pi(theta_opt)(s, a))

                # Normalize by the number of trajectories
                return total_objective / len(sample_trajectories)

            # Optimize the surrogate objective to update policy parameters
            theta = self.optimize(objective, theta)

        # Return the optimized policy parameters after training
        return theta
    
    def sample_trajectories(env, policies, num_trajectories, max_steps=100):
        """
        Collect trajectories by interacting with the environment using the given policy.

        Args:
            env: The environment to interact with.
            policy: The policy function (pi(s, a)).
            num_trajectories: Number of trajectories to collect.
            max_steps: Maximum number of steps per trajectory.

        Returns:
            List of trajectories. Each trajectory is a list of (state, action, reward) tuples.
        """
        custom_keys = ['ongoing_projects', 'completed_projects', 'explored_tiles', 'captured_cities', 'lost_cities', 'enemy_units_eliminated', 'units_lost', 'GDP', 'energy_output', 'resources_controlled', 'environmental_impact']

        trajectories = []

        for _ in range(num_trajectories):

            trajectory = {agent: [] for agent in env.agents}  # Separate trajectory for each agent
            state = env.reset()

            for _ in range(max_steps):
                actions = {}
                for agent in env.agents:
                    obs = env.observe(agent)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    action_logits = policies[agent](obs_tensor)  # Use each agent's policy
                    action_dist = Categorical(logits=action_logits)
                    actions[agent] = action_dist.sample().item()

                    # Take the action in the environment
                    next_obs, rewards, dones, _ = env.step(actions)
                    state_dict = {key: None for key in custom_keys}

                    #TODO

                    """
                    code for getting a dictionary to fill out state_dict

                    state_dict["ongoing_projects"] = 
                    state_dict["completed_projects]=
                    state_dict["explored_tiles"]=
                    state_dict['captured_cities']=
                    state_dict['lost_cities']=
                    state_dict['enemy_units_eliminated']=
                    state_dict['units_lost']=
                    state_dict['GDP']=
                    state_dict['energy_output']=
                    state_dict['resources_controlled']=
                    state_dict['environmental_impact']=
                    """

                    # Store the transition in the trajectory
                    for agent in env.agents:
                        trajectory[agent].append((state_dict, actions[agent], rewards[agent]))


                    # Check if all agents are done
                    if all(dones.values()):
                        break

            trajectories.append(trajectory)

        return trajectories


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

        states = jnp.array(states)
        actions = jnp.array(actions)
        returns = jnp.array(all_returns)

        # Estimate the value function V(s) as the average return for each state
        unique_states = jnp.unique(states, axis=0)
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