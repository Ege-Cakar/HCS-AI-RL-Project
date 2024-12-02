import torch
import numpy as np
from civ import Civilization
from train import ActorRNN, CriticRNN

# Load or define environment and policies
def load_models(num_agents, input_size, hidden_size, output_size):
    """
    Load pre-trained actor and critic models for each agent.
    """
    actor_policies = {}
    critic_policies = {}

    for agent_idx in range(num_agents):
        # Load actor model
        actor = ActorRNN(input_size, hidden_size, output_size)
        actor.load_state_dict(torch.load(f"actor_agent_{agent_idx}.pth"))
        actor.eval()
        actor_policies[f"agent_{agent_idx}"] = actor

        # Load critic model
        critic = CriticRNN(input_size, hidden_size)
        critic.load_state_dict(torch.load(f"critic_agent_{agent_idx}.pth"))
        critic.eval()
        critic_policies[f"agent_{agent_idx}"] = critic

    return actor_policies, critic_policies

def evaluate(env, actor_policies, critic_policies, num_episodes=10, max_steps=100):
    """
    Evaluate the performance of the trained models.

    Args:
        env: The environment to test in.
        actor_policies: Dictionary of actor models for each agent.
        critic_policies: Dictionary of critic models for each agent.
        num_episodes: Number of episodes to test.
        max_steps: Maximum steps per episode.

    Returns:
        results: A dictionary of evaluation metrics.
    """
    results = {
        "total_rewards": [],
        "average_rewards_per_step": [],
        "episode_lengths": [],
    }

    for episode in range(num_episodes):
        state = env.reset()
        actor_hidden_states = {agent: torch.zeros(1, actor_policies[agent].hidden_size) for agent in env.agents}
        critic_hidden_states = {agent: torch.zeros(1, critic_policies[agent].hidden_size) for agent in env.agents}

        episode_rewards = {agent: 0 for agent in env.agents}
        step_count = 0

        for step in range(max_steps):
            actions = {}
            next_actor_hidden_states = {}
            next_critic_hidden_states = {}

            for agent in env.agents:
                obs = env.observe(agent)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

                # Actor decides action
                action_probs, next_actor_hidden = actor_policies[agent](obs_tensor, actor_hidden_states[agent])
                action_dist = torch.distributions.Categorical(probs=action_probs)
                action = action_dist.sample().item()

                actions[agent] = action
                next_actor_hidden_states[agent] = next_actor_hidden

            # Step the environment
            next_state, rewards, dones, _ = env.step(actions)

            # Update episode rewards
            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]

            # Update hidden states
            actor_hidden_states = next_actor_hidden_states

            # Update state
            state = next_state
            step_count += 1

            # If all agents are done, break
            if all(dones.values()):
                break

        # Aggregate episode results
        total_episode_reward = sum(episode_rewards.values())
        results["total_rewards"].append(total_episode_reward)
        results["average_rewards_per_step"].append(total_episode_reward / step_count)
        results["episode_lengths"].append(step_count)

    return results

def main():
    # Initialize the environment
    env = Civilization()  # Assuming Civilization is your environment class

    # Define constants
    num_agents = len(env.agents)
    input_size = env.observation_space.shape[0]  # Example, replace with actual input size
    hidden_size = 128
    output_size = env.action_space.n  # Example, replace with actual action space size

    # Load pre-trained models
    actor_policies, critic_policies = load_models(num_agents, input_size, hidden_size, output_size)

    # Evaluate the models
    results = evaluate(env, actor_policies, critic_policies)

    # Print results
    print(f"Average total rewards over episodes: {np.mean(results['total_rewards'])}")
    print(f"Average rewards per step: {np.mean(results['average_rewards_per_step'])}")
    print(f"Average episode length: {np.mean(results['episode_lengths'])}")

if __name__ == "__main__":
    main()