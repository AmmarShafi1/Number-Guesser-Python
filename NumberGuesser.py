from utils import *
import pygame
from PIL import Image
import numpy as np
from statistics import mode
scale = 14
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Number Guesser")


def init_grid(rows, cols, color):
    grid = []

    for i in range(rows):
        grid.append([])
        for _ in range(cols):
            grid[i].append(color)

    return grid


def draw_grid(win, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            pygame.draw.rect(WIN, pixel, (j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))
    if DRAW_GRID_LINES:
        for i in range(ROWS + 1):
            pygame.draw.line(win, BLACK, (0, i * PIXEL_SIZE), (WIDTH, i * PIXEL_SIZE))
        for i in range(COLS + 1):
            pygame.draw.line(win, BLACK, (i * PIXEL_SIZE, 0), (i * PIXEL_SIZE, HEIGHT - TOOLBAR_HEIGHT))


def draw(win, grid, buttons):
    win.fill(BG_COLOR)
    draw_grid(win, grid)
    for button in buttons:
        button.draw(win)
    show_guess(guess)
    pygame.display.update()


def get_row_col_from_pos(pos):
    x, y = pos
    row = y // PIXEL_SIZE
    col = x // PIXEL_SIZE

    if row >= ROWS:
        raise IndexError

    return row, col


drawing_color = WHITE

run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, BG_COLOR)

button_y = HEIGHT - TOOLBAR_HEIGHT / 2 - 10

buttons = [
    Button(25, button_y, 100, 50, WHITE, "Clear", BLACK),
    Button(130, button_y, 100, 50, WHITE, "Analyze", BLACK),
]

guess = 0
font = pygame.font.Font('freesansbold.ttf', 32)


def show_guess(number):
    guess = font.render("Guess: " + str(number), True, WHITE)
    WIN.blit(guess, (230, button_y))


def get_output(input_data):
    data = np.transpose([input_data])
    return forward_prop(data)


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def forward_prop(X):
    W1 = np.loadtxt(r"W1.out", dtype=float, delimiter=',')
    b1 = np.loadtxt(r"b1.out", dtype=float, delimiter=',')
    W2 = np.loadtxt(r"W2.out", dtype=float, delimiter=',')
    b2 = np.loadtxt(r"b2.out", dtype=float, delimiter=',')
    Z1 = W1.dot(X) + b1
    # A1 is the activated layer
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A2


def process_image(raw_img):
    fixed_img = np.zeros(784)
    i = 0
    for y in range(int(14 / 2), len(raw_img) - 80, scale):
        for x in range(int(14 / 2), len(raw_img[0]), scale):
            fixed_img[i] += (raw_img[y][x][0] / 255)
            i += 1
    # Centers image to prevent mislabels by finding the center of the pixels and attempting to center according to
    # the center of the canvas
    centered_img = np.zeros((28, 28), dtype='int')
    i = 0
    for val in fixed_img:
        centered_img[int(i / 28)][i % 28] = round(val)
        i += 1
    center_x = 0
    center_y = 0
    points = []
    for y in range(28):
        for x in range(28):
            if not centered_img[x][y] == 0:
                center_x += x
                center_y += y
                points.append((x, y))
    center_x /= len(points)
    center_y /= len(points)
    shift_x = center_x - 13.5
    shift_y = center_y - 13.5
    centered_img = np.zeros((28, 28), dtype='int')
    for x, y in points:
        if -1 < x - shift_x < 28 and -1 < y - shift_y < 28:
            centered_img[int(x - shift_x)][int(y - shift_y)] = 1
    i = 0
    # Transferring the centered image back into a single array so we can run it through the neural network
    for arr in centered_img:
        for val in arr:
            fixed_img[i] = val
            i += 1
    return fixed_img


while run:
    clock.tick(FPS)

    for event in pygame.event.get():
        draw(WIN, grid, buttons)
        if event.type == pygame.QUIT:
            run = False
        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            try:
                row, col = get_row_col_from_pos(pos)
                grid[row][col] = drawing_color

            except IndexError:
                for button in buttons:
                    if not button.clicked(pos):
                        continue
                    if button.text == "Clear":
                        grid = init_grid(ROWS, COLS, BG_COLOR)
                    if button.text == "Analyze":
                        pygame.image.save(WIN, "number_guess.jpeg")
                        num_guess = Image.open("number_guess.jpeg", 'r')
                        raw_img = np.array(num_guess)
                        fixed_img = process_image(raw_img)
                        guesses = (np.argmax(get_output(fixed_img), 0))
                        guess = mode(guesses)
                        show_guess(guess)

    show_guess(guess)

pygame.quit()
