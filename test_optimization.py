import time
import numpy as np
from env.civ import Civilization

def test_world_generation():
    print("Testing world generation performance...")
    start_time = time.time()
    
    # Create a Civilization instance with the same map size as your original code
    env = Civilization(map_size=(50, 100), num_agents=2)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"World generation completed in {elapsed:.2f} seconds")
    
if __name__ == "__main__":
    test_world_generation()
