# run_llm_civ.py
"""
Minimal loop: create env, query LLM per AEC turn, step, render.
"""
from env.civ import Civilization
from llm_agent import act
import numpy as np, time
from typing import Any
from llm_agent import _to_jsonable

MAP_SIZE = (40, 60)
NUM_AGENTS = 2

env = Civilization(MAP_SIZE, NUM_AGENTS, render_mode="human")
obs = env.reset()
turn = 0

MAX_UNITS   = env.max_units_per_agent
MAX_CITIES  = env.max_cities
MAX_PROJECTS= env.max_projects
MAP_WIDTH   = env.map_width
MAP_HEIGHT  = env.map_height
NUM_AGENTS  = env.num_of_agents

def build_schema(env):
    props = {
        "action_type":       {"type": "integer", "enum": list(range(13))},
        "unit_id":           {"type": "integer", "enum": list(range(env.max_units_per_agent))},
        "direction":         {"type": "integer", "enum": [0, 1, 2, 3]},
        "city_id":           {"type": "integer", "enum": list(range(env.max_cities))},
        "project_id":        {"type": "integer", "enum": list(range(env.max_projects))},
        "harvest_amount":    {"type": "number"},                 # 0‑1 stated in prompt
        "trade_target":      {"type": "integer", "enum": list(range(env.num_agents))},
        "offer_money":       {"type": "number"},
        "request_money":     {"type": "number"},
        "offer_unit_id":     {"type": "integer", "enum": list(range(env.max_units_per_agent))},
        "request_unit_id":   {"type": "integer", "enum": list(range(env.max_units_per_agent))},
        "invade_x":          {"type": "integer", "enum": list(range(env.map_width))},
        "invade_y":          {"type": "integer", "enum": list(range(env.map_height))},
        "target_government": {"type": "integer", "enum": [0, 1, 2, 3]},
    }

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(props.keys()),
        "properties": props
    }

    return {"type": "json_schema", "json_schema": {
        "name": "action",
        "schema": schema,
        "strict": True
    }}

action_format = build_schema(env)
# Initialize action history dictionary
action_history = {}
for agent_id in env.possible_agents:
    action_history[agent_id] = []

while env.agents:
    agent = env.agent_selection
    observation = env.observe(agent)       # always ask env for current view
    
    # Get action with retry until valid
    valid_action = False
    max_retries = 5  # Maximum number of attempts to get a valid action
    attempt = 0
    
    while not valid_action and attempt < max_retries:
        action = act(observation, action_format, agent, turn, action_history)
        
        if action is not None and env.action_space(agent).contains(action):
            valid_action = True
            # Only add valid actions to the history
            action_history[agent].append(_to_jsonable(action))
            print(f"[INFO] Agent {agent} - Valid action accepted: {action}")
        else:
            if action is None:
                print(f"[WARN] invalid action from LLM; attempt {attempt+1}/{max_retries} - Action is None")
            else:
                print(f"[WARN] invalid action from LLM; attempt {attempt+1}/{max_retries}")
                try:
                    # Check what's wrong with the action space
                    print(f"[DEBUG] Action space sample: {env.action_space(agent).sample()}")
                    print(f"[DEBUG] Action space shape: {env.action_space(agent).shape}")
                    
                    # Try to identify specific validation errors
                    if isinstance(action, dict):
                        for key, value in action.items():
                            if key not in env.action_space(agent).sample():
                                print(f"[DEBUG] Invalid key in action: {key}")
                    else:
                        print(f"[DEBUG] Action is not a dictionary: {type(action)}")
                except Exception as e:
                    print(f"[DEBUG] Error during action space debugging: {e}")
            
            attempt += 1
    
    # If we couldn't get a valid action after max retries, use a fallback
    if not valid_action:
        print(f"[WARN] failed to get valid action after {max_retries} attempts; sampling fallback")
        action = env.action_space(agent).sample()

    env.step(action)                       # PettingZoo AEC step
    turn += 1

    if env.render_mode == "human":
        env.render()

print("Episode done")
