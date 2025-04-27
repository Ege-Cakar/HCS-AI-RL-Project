import pygame
import sys
from env.civ import Civilization
from performance_graph import generate_performance_graph

pygame.init()

# --- UI Settings ---
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

pygame.display.set_caption("Eco-Civilization Setup")
screen = pygame.display.set_mode((600, 400))  # Static GUI size but actual game will be fullscreen or scaled
font = pygame.font.SysFont(None, 32)
big_font = pygame.font.SysFont(None, 48)
clock = pygame.time.Clock()

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 149, 237)

# --- Input Boxes ---
inputs = {
    "# Agents": {"value": "4", "rect": pygame.Rect(250, 60, 100, 32)},
    "Visibility": {"value": "1", "rect": pygame.Rect(250, 110, 100, 32)},
    "Max Projects": {"value": "5", "rect": pygame.Rect(250, 160, 100, 32)},
    "Reward k1": {"value": "100", "rect": pygame.Rect(250, 210, 100, 32)},
    "Penalty gamma": {"value": "0.00001", "rect": pygame.Rect(250, 260, 100, 32)}
}
active_input = None

# --- Button ---
start_button = pygame.Rect(220, 320, 160, 50)

# --- Main Loop ---
WIDTH, HEIGHT = 600, 400
while True:
    screen.fill(BLACK)

    title = big_font.render("Eco-Civilization Setup", True, WHITE)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 10))

    # Draw inputs
    for i, (label, data) in enumerate(inputs.items()):
        lbl_surface = font.render(label + ":", True, WHITE)
        screen.blit(lbl_surface, (60, 65 + i * 50))

        pygame.draw.rect(screen, WHITE if active_input == label else GRAY, data["rect"], 2)
        txt_surface = font.render(data["value"], True, WHITE)
        screen.blit(txt_surface, (data["rect"].x + 5, data["rect"].y + 5))

    # Draw start button
    pygame.draw.rect(screen, BLUE, start_button)
    btn_text = font.render("Start Game", True, WHITE)
    screen.blit(btn_text, (start_button.x + 25, start_button.y + 10))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            active_input = None
            for label, data in inputs.items():
                if data["rect"].collidepoint(event.pos):
                    active_input = label
            if start_button.collidepoint(event.pos):
                # Collect parameters and launch the environment
                params = {
                    key: float(val["value"]) if "." in val["value"] else int(val["value"])
                    for key, val in inputs.items()
                }

                # Dynamically determine map size based on screen resolution
                target_tile_size = 35
                map_width = screen_width // target_tile_size
                map_height = screen_height // target_tile_size

                print("Launching game with:", params)
                pygame.quit()

                env = Civilization(
                    map_size=(map_height, map_width),
                    num_agents=params["# Agents"],
                    visibility_range=params["Visibility"],
                    max_projects=params["Max Projects"],
                    render_mode="human"
                )
                env.k1 = params["Reward k1"]
                env.gamma = params["Penalty gamma"]

                env.reset()
                running = True
                # Initialize money history tracking
                money_history = {agent: [] for agent in env.agents}
                
                while running and env.agents:
                    env.render()
                    current_agent = env.agent_selection
                    action = env.action_space(current_agent).sample() if not env.terminations[current_agent] else None
                    env.step(action)
                    
                    # Record money for each agent after each step
                    for agent in env.agents:
                        if agent in money_history:  # Check if agent still exists
                            money_history[agent].append(env.money[agent])
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                
                # Display performance graph at the end
                pygame.quit()
                generate_performance_graph(env.agents, money_history)
                sys.exit()

        if event.type == pygame.KEYDOWN:
            if active_input:
                if event.key == pygame.K_RETURN:
                    active_input = None
                elif event.key == pygame.K_BACKSPACE:
                    inputs[active_input]["value"] = inputs[active_input]["value"][:-1]
                elif event.unicode.isdigit() or event.unicode == ".":
                    inputs[active_input]["value"] += event.unicode

    pygame.display.flip()
    clock.tick(30)
