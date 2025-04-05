import numpy as np
import random
import pygame
from pygame import VIDEORESIZE
random.seed(55)
np.random.seed(45)

ACTIONS = {
    0: np.array([0, 1], dtype=np.int8),
    1 : np.array([0, -1], dtype=np.int8)
}

COLOR_TO_RGB = {
    "white" : (255, 255, 255),
    "black" : (0, 0, 0),
    "yellow": (255, 255, 0),
    "red"   : (255, 0, 0),
}

COLOR_TO_GRAY = {
    "black"     : 0,
    "mid_gray"  : 128,
    "white"     : 255,
}

WIDTH = 64
HEIGHT = 64


class Paddle:
    def __init__(self, x, y, width, height, speed=3, color="red"):

        # pos = X, Y coordinates from meshgrid: https://numpy.org/doc/2.2/reference/generated/numpy.meshgrid.html
        self.X, self.Y = np.meshgrid(range(x-width//2, x+width//2), range(y-height//2, y+height//2))
        self._init_X, self._init_Y = self.X, self.Y
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color

    def move(self, action):
        next_Y = self.Y + self.speed * action[1]
        # Move only if it's in grid range
        if np.all(next_Y >= 0) and np.all(next_Y < HEIGHT):
            self.Y = next_Y

    def reset(self):
        self.X, self.Y = self._init_X, self._init_Y

class Ball:
    def __init__(self, x, y, radius, speed=3, color="yellow"):
        self._init_center = np.array((x, y))
        self.center = np.array((x, y))
        self.radius = radius
        self.X, self.Y = self.get_circle()

        angle = np.radians(random.uniform(-45, 45))
        self.dir = np.array((np.cos(angle), np.sin(angle))) * random.choice([-1, 1])
        self.speed = speed
        self.color = color

    def reset(self):
        self.center = self._init_center
        angle = np.radians(random.uniform(-45, 45))
        self.dir = np.array((np.cos(angle), np.sin(angle))) * random.choice([-1, 1])
        self.X, self.Y = self.get_circle()

    def move(self):
        self.center = self.center + np.int16(self.speed * self.dir)
        self.X, self.Y = self.get_circle()

    def get_circle(self): # Get the xx, yy coordinates of a circle representation
        xx, yy = np.meshgrid(
            range(self.center[0] - self.radius, self.center[0] + self.radius + 1),
            range(self.center[1] - self.radius, self.center[1] + self.radius + 1)
        )
        circle_mask = (xx - self.center[0]) ** 2 + (yy - self.center[1]) ** 2 <= self.radius ** 2
        yy_idx, xx_idx = np.where(circle_mask)
        x_coords = xx[yy_idx, xx_idx]
        y_coords = yy[yy_idx, xx_idx]
        return x_coords, y_coords # self.X, self.Y

class Pong:
    def __init__(self, paddle_width=2, paddle_height=11, ball_radius=2, render_mode="rgb", level="low"):

        render_modes = ("rgb", "grayscale")
        assert render_mode in render_modes, "Render mode not supported"
        # TODO: allow for grayscale img to implement only [width, height] array

        self.state = np.zeros((WIDTH, HEIGHT, 3), dtype=np.int8)
        self.paddle1 = Paddle(paddle_width*2, HEIGHT//2, paddle_width, paddle_height)
        self.paddle2 = Paddle(WIDTH - paddle_width*2, HEIGHT//2, paddle_width, paddle_height)
        self.ball = Ball(WIDTH//2, HEIGHT//2, radius=ball_radius)

        self.bot = PongPolicy(level) # TODO: select policy based on level
        self.window = None
        self.score = [0, 0]

        self.steps = 0
    def update_state(self):
        arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
        arr[:, :] = COLOR_TO_RGB["black"]  # Init all grid black

        # Fill array with objects at their place
        arr[self.paddle1.X, self.paddle1.Y] =  COLOR_TO_RGB[self.paddle1.color]
        arr[self.paddle2.X, self.paddle2.Y] =  COLOR_TO_RGB[self.paddle2.color]
        arr[self.ball.X, self.ball.Y] =  COLOR_TO_RGB[self.ball.color]

        return arr

    def display_objects(self, scale_x, scale_y):
        # Draw ball
        pygame.draw.circle(
            self.window,
            self.ball.color,
            self.ball.center * np.array([scale_x, scale_y]),
            self.ball.radius * min(scale_x, scale_y)
        )

        # Draw Paddle1
        paddle1_rect = pygame.Rect(
            self.paddle1.X[0][0] * scale_x,
            self.paddle1.Y[0][0] * scale_y,
            self.paddle1.width * scale_x,
            self.paddle1.height * scale_y
        )
        pygame.draw.rect(self.window, self.paddle1.color, paddle1_rect)

        # Draw Paddle2
        paddle2_rect = pygame.Rect(
            self.paddle2.X[0][0] * scale_x,
            self.paddle2.Y[0][0] * scale_y,
            self.paddle2.width  * scale_x,
            self.paddle2.height * scale_y
        )
        pygame.draw.rect(self.window, self.paddle2.color, paddle2_rect)

    def step(self, action):
        reward = 0
        done = False
        self.steps += 1

        self.paddle1.move(ACTIONS[action])
        self.paddle2.move(self.bot.get_action(self.ball, self.paddle2))
        self.ball.move()

        # If it's goal in any side
        if np.any(self.ball.X <= 0) or np.any(self.ball.X >= WIDTH - 1):
            reward = -1 if np.any(self.ball.X <= 0) else 1
            self.score += np.array([0, 1] if np.any(self.ball.X <= 0) else [1, 0])  # Update score
            self.ball.reset(), self.paddle1.reset(), self.paddle2.reset()  # Reset to init positions
            self.state = self.update_state()
            return self.state, reward, True

        hit_walls = np.any(self.ball.Y <= 0) or np.any(self.ball.Y >= HEIGHT - 1)  # Hits side walls
        hit_paddle1 = ( # Hits paddle1
                np.any(np.isin(self.ball.X, self.paddle1.X))
                and np.any(np.isin(self.ball.Y, self.paddle1.Y))
                and self.ball.dir[0] < 0  # Goes in the right direction
        )
        hit_paddle2 = ( # Hits paddle2
                np.any(np.isin(self.ball.X, self.paddle2.X))
                and np.any(np.isin(self.ball.Y, self.paddle2.Y))
                and self.ball.dir[0] > 0  # Goes in the right direction
        )

        if hit_walls or hit_paddle1 or hit_paddle2:
            self.ball.dir = -self.ball.dir  # Reflect direction
        # Update state with the new object locations
        self.state = self.update_state()
        self.render()
        return self.state, reward, done

    def reset(self):
        if self.window is not None:
            pygame.quit()

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            pygame.display.set_caption("Pong")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.reset()
            #if event.type == pygame.VIDEORESIZE:
             #   self.window = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        window_width, window_height = self.window.get_size()
        scale_x = window_width / WIDTH
        scale_y = window_height / HEIGHT

        font = pygame.font.Font('freesansbold.ttf', int(2*scale_x)) # Use default font, size 20
        clock = pygame.time.Clock()
        background = pygame.Surface((WIDTH * scale_x, HEIGHT * scale_y))
        background.fill(COLOR_TO_RGB["black"])

        # Clear screen
        self.window.blit(background, (0, 0))
        # Display objects
        self.display_objects(scale_x, scale_y)
        # Render score
        score_text = font.render(f"Score: {self.score[0]} - {self.score[1]}", True, COLOR_TO_RGB["white"])
        score_pos = (0.2 * WIDTH * scale_x, 0.2 * HEIGHT * scale_y)
        text_rect = score_text.get_rect(center=score_pos)
        self.window.blit(score_text, text_rect)  # Draw score at top-mid

        pygame.display.flip()
        clock.tick(30)

class PongPolicy:
    "Decides an action based on the ball's and paddle's position."
    def __init__(self, level):
        self.level = level

    def get_action(self, ball, paddle):

        paddle_y = np.max(paddle.Y) - paddle.height/2
        dead_zone = 8

        if ball.center[1] > paddle_y + dead_zone:
            return ACTIONS[0]
        elif ball.center[1] < paddle_y - dead_zone:
            return ACTIONS[1]
        else:
            return np.array([0, 0])



