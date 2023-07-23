import pygame
import random

# Game Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
PLAYER_SIZE = 50
OBSTACLE_SIZE = 50
PLAYER_SPEED = 5
OBSTACLE_SPEED = 5

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Player
player_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - PLAYER_SIZE]

# Obstacles
obstacles = []

def add_obstacle():
    new_obstacle_pos = [random.randint(0, SCREEN_WIDTH - OBSTACLE_SIZE), 0]
    obstacles.append(new_obstacle_pos)

def update_obstacles():
    for obstacle in obstacles:
        obstacle[1] += OBSTACLE_SPEED
        if obstacle[1] > SCREEN_HEIGHT:
            obstacles.remove(obstacle)

# Game loop
running = True
while running:
    clock.tick(30)  # Cap at 30 FPS

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Player movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_pos[0] -= PLAYER_SPEED
    if keys[pygame.K_RIGHT]:
        player_pos[0] += PLAYER_SPEED

    # Keep player on screen
    player_pos[0] = max(0, min(player_pos[0], SCREEN_WIDTH - PLAYER_SIZE))

    # Add a new obstacle every 2 seconds
    if pygame.time.get_ticks() % 2000 < 30:
        add_obstacle()

    # Update obstacles
    update_obstacles()

    # Collision detection
    for obstacle in obstacles:
        if (obstacle[0] < player_pos[0] < obstacle[0] + OBSTACLE_SIZE or
            obstacle[0] < player_pos[0] + PLAYER_SIZE < obstacle[0] + OBSTACLE_SIZE) and \
           obstacle[1] + OBSTACLE_SIZE > player_pos[1]:
            running = False

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw player
    pygame.draw.rect(screen, (0, 255, 0), (player_pos[0], player_pos[1], PLAYER_SIZE, PLAYER_SIZE))

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.rect(screen, (255, 0, 0), (obstacle[0], obstacle[1], OBSTACLE_SIZE, OBSTACLE_SIZE))

    # Flip display
    pygame.display.flip()

pygame.quit()