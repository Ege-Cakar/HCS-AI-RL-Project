import matplotlib.pyplot as plt
import numpy as np

def generate_performance_graph(agents, money_history):
    """
    Generate and display a simple performance graph for agents.
    
    Args:
        agents: List of agent IDs
        money_history: Dictionary with money history for each agent
    """
    plt.figure(figsize=(10, 6))
    
    # Create x-axis (turns)
    turns = list(range(1, len(next(iter(money_history.values()))) + 1))
    
    # Plot money for each agent
    for agent_id, data in money_history.items():
        plt.plot(turns, data, label=f'Agent {agent_id}')
    
    plt.title('Agent Performance (Money)')
    plt.xlabel('Turn')
    plt.ylabel('Money')
    plt.legend()
    plt.grid(True)
    
    # Save the graph to a file
    plt.savefig('agent_performance.png')
    
    # Display the graph
    plt.show()
