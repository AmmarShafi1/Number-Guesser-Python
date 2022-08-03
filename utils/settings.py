import pygame

pygame.init()
pygame.font.init()
SCALE = 14
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 255, 0)
GREEN = (0, 0, 255)
GRAY = (56,56,56)

FPS = 60

WIDTH, HEIGHT = (14*28), ((14*28)+80)

ROWS = COLS = 28

TOOLBAR_HEIGHT = HEIGHT - WIDTH

PIXEL_SIZE = WIDTH // COLS

BG_COLOR = BLACK

DRAW_GRID_LINES = True

def get_font(size):
    return pygame.font.SysFont("comicsans", size)