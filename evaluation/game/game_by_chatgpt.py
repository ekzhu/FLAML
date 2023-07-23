
# import pygame
# import random
# import sys
# from pygame.locals import *

# # Initializing Pygame
# pygame.init()

# # Setting up some basic attributes
# SCREENWIDTH, SCREENHEIGHT = 800, 600
# PLAYERWIDTH, PLAYERHEIGHT = 50, 50
# FPS = 60
# PLAYER_SPEED = 5
# ALIEN_SPEED = 3
# BULLET_SPEED = 10
# ALIEN_PROBABILITY = 0.01

# # Colors
# WHITE = (255, 255, 255)
# GREEN = (0, 255, 0)
# RED = (255, 0, 0)

# # Setting up assets
# PLAYER_IMAGE = pygame.image.load('player_image.png')
# ALIEN_IMAGE = pygame.image.load('alien_image.png')
# BULLET_IMAGE = pygame.image.load('bullet_image.png')

# # Setting up the display
# WINDOW = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))

# def draw_object(image, x, y):
#     WINDOW.blit(image, (x, y))

# def player(x, y):
#     draw_object(PLAYER_IMAGE, x, y)

# def alien(x, y):
#     draw_object(ALIEN_IMAGE, x, y)

# def bullet(x, y):
#     draw_object(BULLET_IMAGE, x, y)

# def isCollision(alienX, alienY, bulletX, bulletY):
#     distance = ((alienX - bulletX) ** 2) + ((alienY - bulletY) ** 2)
#     if distance < 27:
#         return True
#     else:
#         return False

# def game_over():
#     pygame.quit()
#     sys.exit()

# def game_loop():
#     player_x = SCREENWIDTH // 2
#     player_y = SCREENHEIGHT - PLAYERHEIGHT - 10

#     aliens = []
#     bullets = []

#     clock = pygame.time.Clock()

#     while True:
#         WINDOW.fill((0, 0, 0))
#         for event in pygame.event.get():
#             if event.type == QUIT:
#                 game_over()

#             # Player movement
#             if event.type == KEYDOWN:
#                 if event.key == K_LEFT:
#                     player_x -= PLAYER_SPEED
#                 elif event.key == K_RIGHT:
#                     player_x += PLAYER_SPEED

#             # Shooting bullets
#             if event.type == KEYDOWN:
#                 if event.key == K_SPACE:
#                     bullets.append([player_x, player_y])

#         # Alien generation
#         if len(aliens) < 10 and random.random() < ALIEN_PROBABILITY:
#             aliens.append([random.randint(0, SCREENWIDTH), 0])

#         # Draw & update bullet positions
#         for b in bullets:
#             bullet(*b)
#             b[1] -= BULLET_SPEED

#         # Draw & update alien positions
#         for a in aliens:
#             alien(*a)
#             a[1] += ALIEN_SPEED

#         # Collision detection
#         for a in aliens:
#             if a[1] > SCREENHEIGHT:
#                 game_over()
#             for b in bullets:
#                 if isCollision(*a, *b):
#                     aliens.remove(a)
#                     bullets.remove(b)
#                     break

#         # Draw player
#         player(player_x, player_y)

#         pygame.display.update()
#         clock.tick(FPS)

# if __name__ == '__main__':
#     game_loop()


import pygame
import random
import sys
from pygame.locals import *

# Initializing Pygame
pygame.init()

# Setting up some basic attributes
SCREENWIDTH, SCREENHEIGHT = 800, 600
PLAYERWIDTH, PLAYERHEIGHT = 50, 50
FPS = 60
PLAYER_SPEED = 5
ALIEN_SPEED = 3
BULLET_SPEED = 10
ALIEN_PROBABILITY = 0.01

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Setting up the display
WINDOW = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))

def player(x, y):
    pygame.draw.rect(WINDOW, GREEN, pygame.Rect(x, y, PLAYERWIDTH, PLAYERHEIGHT))

def alien(x, y):
    pygame.draw.circle(WINDOW, RED, (x, y), 20)

def bullet(x, y):
    pygame.draw.line(WINDOW, WHITE, (x, y), (x, y+10), 2)

def isCollision(alienX, alienY, bulletX, bulletY):
    distance = ((alienX - bulletX) ** 2) + ((alienY - bulletY) ** 2)
    if distance < 27:
        return True
    else:
        return False

def game_over():
    pygame.quit()
    sys.exit()

def game_loop():
    player_x = SCREENWIDTH // 2
    player_y = SCREENHEIGHT - PLAYERHEIGHT - 10

    aliens = []
    bullets = []

    clock = pygame.time.Clock()

    while True:
        WINDOW.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == QUIT:
                game_over()

            # Player movement
            if event.type == KEYDOWN:
                if event.key == K_LEFT:
                    player_x -= PLAYER_SPEED
                elif event.key == K_RIGHT:
                    player_x += PLAYER_SPEED

            # Shooting bullets
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    bullets.append([player_x, player_y])

        # Alien generation
        if len(aliens) < 10 and random.random() < ALIEN_PROBABILITY:
            aliens.append([random.randint(0, SCREENWIDTH), 0])

        # Draw & update bullet positions
        for b in bullets:
            bullet(*b)
            b[1] -= BULLET_SPEED

        # Draw & update alien positions
        for a in aliens:
            alien(*a)
            a[1] += ALIEN_SPEED

        # Collision detection
        for a in aliens:
            if a[1] > SCREENHEIGHT:
                game_over()
            for b in bullets:
                if isCollision(*a, *b):
                    aliens.remove(a)
                    bullets.remove(b)
                    break

        # Draw player
        player(player_x, player_y)

        pygame.display.update()
        clock.tick(FPS)

if __name__ == '__main__':
    game_loop()
