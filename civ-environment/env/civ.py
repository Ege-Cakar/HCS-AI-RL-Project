import numpy as np
import pettingzoo as pz
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv
import pygame
from pygame.locals import QUIT
import math

class Civilization(AECEnv): 
    metadata = {'render.modes': ['human'], 'name': 'Civilization_v0'}
    def __init__(self, map_size, num_agents, max_cities=10, visibility_range=1, *args, **kwargs):
        """
        Initialize the Civilization game.
        Args:
            map_size (tuple): The size of the map (width, height).
            num_agents (int): The number of players in the game.
            max_cities (int): Maximum number of cities per agent.
            visibility_range (int): The range of visibility for each unit (the tiles the units are on, and tiles within borders are already included).
            *args: Additional positional arguments for the parent class.
            **kwargs: Additional keyword arguments for the parent class.
        """
        """
        Each agent can have cities, warrios and settlers. For a maximum of 6 agents, that's 18 slots occupied. 
        Each agent can also see resources, materials, and water. That's 3 more. 
        """

        super().__init__()
        if num_agents > 6:
            raise ValueError(
                f"Number of players ({num_agents}) exceeds the maximum allowed (6)."
            )
        self.agents = ["player_" + str(i) for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.current_agent = self.agent_selector.reset()
        self.num_of_agents = num_agents

        self.map_size = map_size
        self.map_height, self.map_width = map_size

        self.max_cities = max_cities
        # Initialize the observation spaces for each player
        self.observation_spaces = {
            agent: spaces.Dict({
                "map": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(
                        self.map_height, 
                        self.map_width, 
                        self._calculate_num_channels()
                    ), 
                    dtype=np.float32
                ),
                "cities": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(
                        self.max_cities,
                        self._calculate_city_attributes()
                    ),
                    dtype=np.float32
                )
            })
            for agent in self.agents
        }
        # Initialize the action spaces for each player
        self.action_spaces = {
            agent: spaces.Discrete(10)  # Will change, placeholder for now
            for agent in self.agents
        }
       
        self.visibility_range = visibility_range
        self._initialize_map

        # Initialize Pygame:
        pygame.init()
        self.cell_size = 40  # Size of each tile in pixels
        self.window_width = self.map_width * self.cell_size
        self.window_height = self.map_height * self.cell_size
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Civilization Environment')
        self.clock = pygame.time.Clock()
        #This can definitebly be improved, but for now it's just a placeholder.
        #This is straught from the internet, needs to be changed
    
    def _calculate_num_channels(self):
        """
        Calculate the number of channels needed for the map representation, which changes dynamically based on number of players.
        """
        ownership_channels = self.num_agents  # One channel per agent for ownership
        units_channels = 3 * self.num_agents  # Cities, Warriors, Settlers per player
        resources_channels = 3  # Resources, Materials, Water
        return ownership_channels + units_channels + resources_channels

    def _calculate_city_attributes(self):
        """
        Calculate the number of attributes per city.
        Attributes:
            - Health
            - X location
            - Y location
            - Resources
            - Finished Projects (one-hot for each possible project)
            - Current Project
            - Project Duration
        """
        num_projects = 5  # Placeholder, needs to change
        return 1 + 2 + 3 + num_projects + 1 + 1  # Health, Location (x, y), Resources(3, 1 for each type), Finished Projects, Current Project, Duration

    
    def _initialize_map(self, seed=None):
        """
        Initialize the map with zeros or default values, place resources, and set spawn points for settlers warriors.
        Args:
            seed (int, optional): Seed for the random number generator to ensure reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        
        num_channels = self._calculate_num_channels()
        self.map_height, self.map_width = self.map_size
        self.map = np.zeros((self.map_height, self.map_width, num_channels), dtype=np.float32)
    
        # Randomly place resources on the map
        self._place_resources()
    
        # Place spawn settlers and warriors for each player
        self._place_starting_units()
    
    def _place_resources(self, bountifulness=0.15):
        """
        Randomly place resources, materials, and water on the map.
        """
        num_resources = int(bountifulness * self.map_height * self.map_width)
        resource_channels_start = self.num_agents + 3 * self.num_agents  # Starting index for resource channels, since this much will be occupied by borders and units

        # Channels for resources
        resources_channel = resource_channels_start  # Index for energy resources
        materials_channel = resource_channels_start + 1  # Index for materials
        water_channel = resource_channels_start + 2  # Index for water

        all_tiles = [(x, y) for x in range(self.map_width) for y in range(self.map_height)]
        np.random.shuffle(all_tiles)  # Shuffle the list to randomize tile selection
        # POSSIBLE BOTTLENECK!

        resources_placed = 0
        tile_index = 0

        while resources_placed < num_resources and tile_index < len(all_tiles):
            x, y = all_tiles[tile_index]
            tile_index += 1

            # Check if there is already a resource on this tile
            tile_resources = self.map[y, x, resources_channel:water_channel + 1]
            if np.any(tile_resources > 0):
                continue  

            # Randomly choose a resource type to place
            resource_type = np.random.choice(['resource', 'material', 'water'])
            if resource_type == 'resource':
                self.map[y, x, resources_channel] = 1
            elif resource_type == 'material':
                self.map[y, x, materials_channel] = 1
            elif resource_type == 'water':
                self.map[y, x, water_channel] = 1

            resources_placed += 1
        
    def _place_starting_units(self):
        """
        Place spawn points for settlers and starting units (e.g., warriors) for each player.
        """
        spawn_points = []
        for agent_idx in range(self.num_agents):
            while True:
                x = np.random.randint(0, self.map_width)
                y = np.random.randint(0, self.map_height)
                # Ensure the tile is empty (and not too close to other spawn points?)
                if self._is_tile_empty(x, y):
                    break
            spawn_points.append((x, y))
            self._place_unit(agent_idx, 'settler', x, y)
            adjacent_tiles = self._get_adjacent_tiles(x, y) # Put in the first possible tile
            warrior_placed = False
            for adj_x, adj_y in adjacent_tiles:
                if self._is_tile_empty(adj_x, adj_y):
                    # Place the warrior at (adj_x, adj_y)
                    self._place_unit(agent_idx, 'warrior', adj_x, adj_y)
                    warrior_placed = True
                    break
            if not warrior_placed:
                # Handle the case where no adjacent empty tile is found
                print(f"Warning: Could not place warrior for agent {agent_idx} adjacent to settler at ({x}, {y}).")
                # Optionally, expand search radius

    
    def _is_tile_empty(self, x, y):
        """
        Check if a tile is empty (no units, resources, or ownership).
        # TODO: Right now, this is **too** simple. It's fine if there are resources, just need to make sure it's not owned and there are no other units.
        It might be a good idea to make this return as to *what* is there (nothing, unit from player 2 + resource, etc.) and go on with that information.
        """
        return np.all(self.map[y, x, :] == 0)

    def _place_unit(self, agent_idx, unit_type, x, y):
        """
        Place a unit of a specific type for a given agent at the specified location.
        Args:
            agent_idx: Index of the agent.
            unit_type: 'city', 'warrior', or 'settler'.
            x, y: Coordinates to place the unit.
        """
        unit_types = {'city': 0, 'warrior': 1, 'settler': 2}
        if unit_type not in unit_types:
            raise ValueError(f"Invalid unit type: {unit_type}") #no typos!
        unit_channel = self.num_agents + (3 * agent_idx) + unit_types[unit_type]
        self.map[y, x, unit_channel] = 1
    
    def _get_adjacent_tiles(self, x, y):
        """
        Get a list of adjacent tile coordinates to (x, y), considering map boundaries.
        TODO: Add a check for units of other players, to utilize this for attacking etc. as well. 
        """
        adjacent_coords = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # This is just the root tile
                adj_x, adj_y = x + dx, y + dy
                # Check if the adjacent tile is within map boundaries
                if 0 <= adj_x < self.map_width and 0 <= adj_y < self.map_height:
                    adjacent_coords.append((adj_x, adj_y))
        return adjacent_coords

    def render(self):
        """
        Visualize the current state of the map using Pygame.
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        # Background
        self.screen.fill((0, 0, 0))  # Black background

        # Draw the grid and elements
        self._draw_grid()
        self._draw_elements()

        pygame.display.flip()
        self.clock.tick(60)  # Limit to 60 fps
        
    def _draw_grid(self):
        """
        Draw the grid lines on the screen.
        """
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.window_width, y))
    
    def _draw_elements(self):
        """
        Draw the settlers, warriors, ownership, and resources on the map.
        """
        # Define colors
        agent_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Cyan
        ]
        resource_colors = {
            'resource': (200, 200, 200),   # Light gray
            'material': (139, 69, 19),     # Brown
            'water': (0, 191, 255)         # Deep sky blue
        }
        # Draw ownership (background color of tiles)
        for y in range(self.map_height):
            for x in range(self.map_width):
                for agent_idx in range(self.num_agents):
                    if self.map[y, x, agent_idx] == 1:
                        color = agent_colors[agent_idx % len(agent_colors)]
                        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                        pygame.draw.rect(self.screen, color, rect)
                        break  # Only one player can own a tile
        # Draw resources
        resource_channels_start = self.num_agents + 3 * self.num_agents
        resources_channel = resource_channels_start
        materials_channel = resource_channels_start + 1
        water_channel = resource_channels_start + 2
        for y in range(self.map_height):
            for x in range(self.map_width):
                # Resources
                if self.map[y, x, resources_channel] == 1:
                    self._draw_circle(x, y, resource_colors['resource'])
                if self.map[y, x, materials_channel] == 1:
                    self._draw_circle(x, y, resource_colors['material'])
                if self.map[y, x, water_channel] == 1:
                    self._draw_circle(x, y, resource_colors['water'])
        # Draw units
        for agent_idx in range(self.num_agents):
            unit_base_idx = self.num_agents + (3 * agent_idx)
            city_channel = unit_base_idx + 0    # 'city'
            warrior_channel = unit_base_idx + 1  # 'warrior'
            settler_channel = unit_base_idx + 2  # 'settler'
            # Cities
            city_positions = np.argwhere(self.map[:, :, city_channel] == 1)
            # Make the city color slightly darker than the agent color
            darker_color = tuple(
                max(0, min(255, int(c * 0.7))) for c in agent_colors[agent_idx % len(agent_colors)]
            )
            for y_pos, x_pos in city_positions:
                self._draw_star(x_pos, y_pos, darker_color)
            # Warriors
            warrior_positions = np.argwhere(self.map[:, :, warrior_channel] == 1)
            for y_pos, x_pos in warrior_positions:
                self._draw_triangle(x_pos, y_pos, agent_colors[agent_idx % len(agent_colors)])
            # Settlers
            settler_positions = np.argwhere(self.map[:, :, settler_channel] == 1)
            for y_pos, x_pos in settler_positions:
                self._draw_square(x_pos, y_pos, agent_colors[agent_idx % len(agent_colors)])

    def _draw_circle(self, x, y, color):
        """
        Draw a circle (resource) at the given map coordinates.
        """
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 4
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)

    def _draw_square(self, x, y, color):
        """
        Draw a square (settler) at the given map coordinates. # Placeholder
        """
        padding = self.cell_size // 8
        rect = pygame.Rect(
            x * self.cell_size + padding,
            y * self.cell_size + padding,
            self.cell_size - 2 * padding,
            self.cell_size - 2 * padding
        )
        pygame.draw.rect(self.screen, color, rect)

    def _draw_triangle(self, x, y, color):
        """
        Draw a triangle (warrior) at the given map coordinates.
        """
        half_size = self.cell_size // 2
        quarter_size = self.cell_size // 4
        center_x = x * self.cell_size + half_size
        center_y = y * self.cell_size + half_size
        points = [
            (center_x, center_y - quarter_size),  # Top point
            (center_x - quarter_size, center_y + quarter_size),  # Bottom left
            (center_x + quarter_size, center_y + quarter_size)   # Bottom right
        ]
        pygame.draw.polygon(self.screen, color, points)

    def _draw_star(self, x, y, color):
        """
        Draw a star (city) at the given map coordinates.
        """
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        radius_outer = self.cell_size // 3
        radius_inner = self.cell_size // 6
        num_points = 5
        points = []
        for i in range(num_points * 2):
            angle = i * math.pi / num_points - math.pi / 2  # Rotate to point upwards
            if i % 2 == 0:
                r = radius_outer
            else:
                r = radius_inner
            px = center_x + r * math.cos(angle)
            py = center_y + r * math.sin(angle)
            points.append((px, py))
        pygame.draw.polygon(self.screen, color, points)
    # polygon code from the internet
    def reset(self):
        """
        Reset the environment.
        """
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.current_agent = self.agent_selector.next()
        self._initialize_map()
        # Reset rewards, done, and info
        return #Observation of the current agent


# Testing 
if __name__ == "__main__":
    map_size = (10, 10) 
    num_agents = 2        
    env = Civilization(map_size, num_agents)
    env.reset()
    running = True
    while running:
        env.render()
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
    pygame.quit()