import pygame
import sys

pygame.init()

# --- UI Settings ---
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Eco-Civilization Setup")
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
    "Map Width": {"value": "40", "rect": pygame.Rect(250, 60, 100, 32)},
    "Map Height": {"value": "30", "rect": pygame.Rect(250, 110, 100, 32)},
    "# Agents": {"value": "4", "rect": pygame.Rect(250, 160, 100, 32)},
    "Visibility": {"value": "1", "rect": pygame.Rect(250, 210, 100, 32)}
}
active_input = None

# --- Button ---
start_button = pygame.Rect(220, 280, 160, 50)

# --- Main Loop ---
while True:
    screen.fill(BLACK)

    # Title
    title = big_font.render("🌱 Eco-Civilization Setup", True, WHITE)
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

    # Event handling
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
                params = {k: int(v["value"]) for k, v in inputs.items()}
                print("Launching game with:", params)
                pygame.quit()
                # import and launch game here, e.g.,
                from civ import Civilization
                env = Civilization(
                    map_size=(params["Map Height"], params["Map Width"]),
                    num_agents=params["# Agents"],
                    visibility_range=params["Visibility"],
                    render_mode="human"
                )
                env.reset()
                running = True
                while running and env.agents:
                    env.render()
                    current_agent = env.agent_selection
                    action = env.action_space(current_agent).sample() if not env.terminations[current_agent] else None
                    env.step(action)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                sys.exit()

        if event.type == pygame.KEYDOWN:
            if active_input:
                if event.key == pygame.K_RETURN:
                    active_input = None
                elif event.key == pygame.K_BACKSPACE:
                    inputs[active_input]["value"] = inputs[active_input]["value"][:-1]
                elif event.unicode.isdigit():
                    inputs[active_input]["value"] += event.unicode

    pygame.display.flip()
    clock.tick(30)
