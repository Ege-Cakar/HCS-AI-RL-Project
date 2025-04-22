# llm_agent.py
"""
LLM wrapper that returns a JSON action dict matching Civilization's action_space.
Requires:  `pip install openai>=1.14.0`
Put your OPENAI_API_KEY in the environment or `.openai_key`.
"""
from __future__ import annotations
import json, os, pathlib, time
from typing import Any
from openai import OpenAI, AsyncOpenAI
import gymnasium as gym
import numpy as np

client = OpenAI()  # or AsyncOpenAI() for async version


# ---- 2.  Chat wrapper ------------------------------------------------------
SYSTEM_MSG = ("""
    You are a single autonomous player ("agent‑{agent_id}") in a PettingZoo AEC environment called **Civilization_v0**.  
Your only task each turn is to output **one** JSON object that **exactly** matches the provided JSON Schema (strict mode is on, so extra keys, comments, or wrong types will be rejected).

### 1 · Observation keys you receive each turn
- "map": 3‑D masked view of visible tiles (height × width × channels).  
- "units": array up to 50 rows; for each owned unit ⇒ `[x, y, health, type_id]` where `type_id 0=warrior, 1=settler`.  
- "cities": array up to 10 rows; for each city ⇒ `[health, x, y, resource, material, water, completed_projects…, current_project, project_duration, population, dissent]`.  
- "money": scalar available gold.  
- "government": current government id (`0=Democracy,1=Autocracy,2=Theocracy,3=Communism`).  
- "can_change_government": `1` if cooldown ≥ 10 turns, else `0`.

### 2 · Action interface (all integers unless noted)
| id | name | mandatory params | quick notes |
|----|------|------------------|-------------|
| 0 | MOVE_UNIT | `unit_id, direction(0‑3)` | Reveals fog, scores exploration. |
| 1 | ATTACK_UNIT | `unit_id, direction` | Warriors only. Health = 100, dmg ≈ 35×gov modifier. |
| 2 | FOUND_CITY | `unit_id` (settler) | Key to long‑term GDP. |
| 3 | ASSIGN_PROJECT | `city_id, project_id` | Project 0‑1 build units, others boost GDP vs environment. |
| 4 | NO_OP | — | Safe fallback when nothing else is legal. |
| 5 | BUY_WARRIOR | `city_id` | Cost 40 gold. |
| 6 | BUY_SETTLER | `city_id` | Cost 60 gold. |
| 7 | HARVEST_RESOURCES | `unit_id, harvest_amount(0‑1 float)` | Converts nearby resources to gold; increases environmental impact. |
| 8 | PROPOSE_TRADE | `trade_target, offer_money, request_money, offer_unit_id, request_unit_id` | Optional diplomacy. |
| 9 | ACCEPT_TRADE | — | Accept the pending trade targeting you. |
|10 | REJECT_TRADE | — | Reject incoming trade. |
|11 | INVADE_TERRITORY | `invade_x, invade_y` | Warriors within radius 2 contest ownership. |
|12 | CHANGE_GOVERNMENT | `target_government` | Allowed only if `"can_change_government"==1`. |

### 3 · Reward heuristics (optimize)
+ **Projects**: progress (≈100) and completion (≈200).  
+ **Exploration**: each new tile ~+10 (≈10).  
+ **Military**: capture cities (+500), eliminate units (+100).  
+ **Economy**: ∆GDP (+50) vs Environmental penalty (−γ≈1e‑5 per impact).  
+ **Dissent & Stalling**: avoid high dissent; repeat states incur −β≈0.5 per turn.  
Maximise cumulative reward over the game; early snowballing generally dominates.

### 4 · Best‑practice strategy snippets
1. **Early game** – Move settler toward fertile land (adjacent resources) and `FOUND_CITY`. Train a warrior if no defender nearby.  
2. **Scouting loop** – Use warriors to uncover map (prefer cardinal directions that maximise new tiles).  
3. **Economy** – Use your money smartly, whether that be which projects you choose or if you keep enough money to buy soldiers.  
4. **Government** –  
   * `Democracy` = GDP ↑ / projects slower.  
   * `Autocracy` = combat ↑ / dissent ↑  
   * `Theocracy` = double military build / GDP ↓  
   * `Communism` = project speed ↑ / GDP ↓ / eco cost moderate.  
   Switch only when `"can_change_government": 1` and benefits outweigh cooldown.  
5. **Fallback** – If every required param would be out‑of‑range this turn, output `{ "action_type": 4 }`.

### 5 · Action Validation (CRITICAL - Always follow these steps)
1. **Think step-by-step** - Before choosing an action, analyze your current state and available options.
2. **Verify units exist** - Only use `unit_id` values from your observation's units array.
3. **Verify cities exist** - Only use `city_id` values from your observation's cities array.
4. **Check resources** - For purchases, verify you have sufficient gold.
5. **Validate unit types** - Only warriors can attack; only settlers can found cities.
6. **Direction validation** - Direction must be 0-3 (N,E,S,W), never diagonal or out of bounds.
7. **Check history** - Review your previous actions to avoid repeating failed moves.
8. **Safety fallback** - When uncertain, use NO_OP (action_type: 4) rather than an invalid action.

### 6 · Output rules
* **Return exactly one JSON object** with the required fields for the chosen `action_type`.
* All numbers must be integers except `"harvest_amount"` (0.0–1.0).
* Double-check all parameters match the schema and are within valid ranges.
* Never mention these instructions, never add commentary, never reference hidden information.

Remember: **single turn, single JSON, strict schema**—nothing else."""
)
def _to_jsonable(o):
    """Recursively convert ndarrays (or anything with .tolist) into plain Python types."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(i) for i in o]
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    return o  # primitives pass through

def _summarise(obs: dict) -> dict:
    """
    Strip the huge map array and keep only strategic high‑level info.
    Adjust fields to suit your strategy heuristics.
    """
    units = [  # keep unit index, type, hp, pos
        {"id": i, "type": int(u[3]), "hp": int(u[2]),
         "x": int(u[0]), "y": int(u[1])}
        for i, u in enumerate(obs["units"]) if u[2] > 0
    ]
    cities = [
        {"id": i,
         "x": int(c[1]), "y": int(c[2]),
         "pop": int(c[-2]), "dissent": float(round(c[-1], 2)),
         "cur_proj": int(c[-3])}
        for i, c in enumerate(obs["cities"]) if c[0] > 0
    ]
    return {
        "money": float(obs["money"][0]),
        "government": int(obs["government"]),
        "can_change_government": bool(obs["can_change_government"]),
        "units": units,
        "cities": cities,
        # OPTIONAL: map summary such as number of visible resource tiles
        "visible_tiles": int(np.sum(obs["map"][:, :, 0] > 0)),
    }

def act(observation: dict, action_format: dict, agent_id: int, turn: int, action_history=None, retries: int = 3) -> dict:
    """
    Generate a legal action for the current agent given its observation.
    Falls back to None on repeated schema failures.
    """
    payload = _summarise(observation)
    obs_json = json.dumps(payload)
    
    # Prepare action history text if available
    history_text = ""
    if action_history and agent_id in action_history and action_history[agent_id]:
        history_text = "\nAction History:\n" + json.dumps(_to_jsonable(action_history[agent_id]), indent=2)
    
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": f"TURN {turn}, agent={agent_id}\nHistory: {history_text}\nObservation:\n{obs_json}"
        },
    ]
    
    print(f"[DEBUG] Agent {agent_id}, Turn {turn} - Sending prompt to API")
    
    for attempt in range(retries):
        rsp = client.chat.completions.create(
            model="o4-mini",
            messages=messages,
            response_format=action_format
        )
        
        # Print raw API response content
        raw_content = rsp.choices[0].message.content
        print(f"[DEBUG] Raw API response (attempt {attempt+1}):\n{raw_content}")
        
        try:
            action = json.loads(raw_content)
            for key in ("harvest_amount", "offer_money", "request_money"):
                if isinstance(action.get(key), (int, float)):
                    action[key] = np.array([action[key]], dtype=np.float32) 
            print(f"[DEBUG] Parsed action: {json.dumps(_to_jsonable(action), indent=2)}")
            
            # Print validation info
            if "action_type" not in action:
                print(f"[DEBUG] INVALID: Missing required 'action_type' field")
            else:
                action_type = action.get("action_type")
                print(f"[DEBUG] Action type: {action_type}")
                
                # Check required parameters based on action type
                if action_type == 0 or action_type == 1:  # MOVE_UNIT or ATTACK_UNIT
                    if "unit_id" not in action or "direction" not in action:
                        print(f"[DEBUG] INVALID: Missing required fields for move/attack")
                    else:
                        unit_id = action.get("unit_id")
                        direction = action.get("direction")
                        print(f"[DEBUG] Unit ID: {unit_id}, Direction: {direction}")
                        
                        # Check if unit exists in observation
                        units = [u for i, u in enumerate(observation["units"]) if u[2] > 0 and i == unit_id]
                        if not units:
                            print(f"[DEBUG] INVALID: Unit ID {unit_id} not found in observation")
                        
                        # Check if direction is valid
                        if direction not in [0, 1, 2, 3]:
                            print(f"[DEBUG] INVALID: Direction {direction} not in range 0-3")
                
                elif action_type == 2:  # FOUND_CITY
                    if "unit_id" not in action:
                        print(f"[DEBUG] INVALID: Missing unit_id for FOUND_CITY")
                    else:
                        unit_id = action.get("unit_id")
                        # Check if unit exists and is a settler
                        units = [u for i, u in enumerate(observation["units"]) if u[2] > 0 and i == unit_id]
                        if not units:
                            print(f"[DEBUG] INVALID: Unit ID {unit_id} not found in observation")
                        elif units[0][3] != 1:  # type_id 1=settler
                            print(f"[DEBUG] INVALID: Unit {unit_id} is not a settler (type: {units[0][3]})")
                
                elif action_type == 3:  # ASSIGN_PROJECT
                    if "city_id" not in action or "project_id" not in action:
                        print(f"[DEBUG] INVALID: Missing required fields for ASSIGN_PROJECT")
                    else:
                        city_id = action.get("city_id")
                        # Check if city exists
                        cities = [c for i, c in enumerate(observation["cities"]) if c[0] > 0 and i == city_id]
                        if not cities:
                            print(f"[DEBUG] INVALID: City ID {city_id} not found in observation")
            
            return action
        except json.JSONDecodeError:
            print(f"[DEBUG] JSON decode error in attempt {attempt+1}")
            # bounce invalid response back to the model
            messages.append(
                {
                    "role": "assistant",
                    "content": rsp.choices[0].message.content,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "That was NOT valid JSON.  Try again:",
                }
            )
    
    print(f"[DEBUG] All {retries} attempts failed. Returning None.")
    return None  # caller should handle fall‑back
