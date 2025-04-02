import numpy as np
from enum import IntEnum
import random

from matplotlib.pyplot import xscale


class Objects(IntEnum):
    empty = 0
    wall = 1
    paddle = 2
    ball = 3

actions = {
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
    "black"     : (0, 0, 0),
    "dark_gray"  : (50, 50, 50),
    "mid_gray"  : (128, 128, 128),
    "light_gray": (200, 200, 200),
    "white"     : (255, 255, 255),
}

class Paddle:
    def __init__(self, x, y, width=10, height=30):

        #self.shape = width, height
        # TODO: self.pos has to be a 2D with all the positions filled by the Paddle
        # [x: , y: ] * length [0, i] cause it's vertical so j will be the same
        # pos = X, Y coordinates from meshgrid: https://numpy.org/doc/2.2/reference/generated/numpy.meshgrid.html
        self.X, self.Y = np.meshgrid(range(x-width//2, x+width//2), range(y-height//2, y+height//2), indexing="ij")
        self.color = "red"

class Ball:
    def __init__(self, x, y, size = 10):
        self.X, self.Y = np.meshgrid(range(x-size//2, x+size//2), range(y-size//2, y+size//2), indexing="ij")
        self.dir_space = np.array([[x, y] for y in range(-1, 2) for x in range(-1, 2) if x!=0 or y!=0]) # dir R^2 -> {-1, 0, 1} X {-1, 0, 1} / {0, 0}
        self.dir = np.array(random.choice(self.dir_space))
        self.color = "yellow"

class Pong:
    def __init__(self, width, height, paddle_width=None, paddle_height=None, ball_size=None):
        self.width = width
        self.height = height
        self.state = np.zeros((width, height), dtype=np.int8)
        self.paddle1 = Paddle(10, height//2)
        self.paddle2 = Paddle(width-10, height//2)
        self.ball = Ball(width//2, height//2, )

    def step(self, action):
        reward = 0
        done = False

        def in_grid(X, Y): # : 2D array np.all(pos[0] >= 0 and pos[0] < a.shape[0]) and np.all(pos[1] >= 0 and pos[1] < a.shape[1])
            return (np.all(0 <= X) and np.all(X < self.width) and
                    np.all(0 <= Y) and np.all(Y < self.height))
            #return True if 0 <= pos[0] < self.state.shape[0] and 0 <= pos[1] < self.state.shape[1] else False

        paddle1_next_X = self.paddle1.X + actions[action][0]
        paddle1_next_Y = self.paddle1.Y + actions[action][1]
        paddle2_next_X = self.paddle2.X + actions[action][0]
        paddle2_next_Y = self.paddle2.Y + actions[action][1]

        if not in_grid(paddle1_next_X, paddle1_next_Y): # any of the edges outside the grid
            return self.state, reward, done

        # TODO: update ball position
        ball_next_X = self.ball.X + self.ball.dir[0]
        ball_next_Y = self.ball.Y + self.ball.dir[1]

        # If ball coincides in X and Y
        if np.any(ball_next_X < 0) or np.any(ball_next_X > self.state.shape[0]): # It means Goal
            reward = -1 if ball_next_X < 0 else 1
            return self.state, reward, True

        if (
                not in_grid(ball_next_X, ball_next_Y) # Outside thre grid
                or np.any(np.isin(ball_next_X, paddle1_next_X)) and np.any(np.isin(ball_next_Y, paddle1_next_Y)) # Hits paddle1
                or np.any(np.isin(ball_next_X, paddle2_next_X)) and np.any(np.isin(ball_next_Y, paddle2_next_Y)) # Hits paddle2
        ):
            self.ball.dir = -self.ball.dir
            self.ball.pos = self.ball.pos + self.ball.dir

        # Update paddle positions
        self.paddle1.X, self.paddle1.Y  = paddle1_next_X, paddle1_next_Y
        self.paddle2.X, self.paddle2.Y  = paddle2_next_X, paddle2_next_Y

        return self.state, reward, done

    def reset(self):
        ...

    def render(self):

        import matplotlib.pyplot as plt
        from PIL import Image

        arr = np.zeros((self.state.shape[0], self.state.shape[1], 3), dtype=np.uint8)
        arr[:, :] = COLOR_TO_RGB["black"] # Init all grid black

        for obj in [self.paddle1, self.paddle2, self.ball]:
            arr[obj.X, obj.Y] = COLOR_TO_RGB[obj.color]

        import pygame

        # Initialize Pygame
        pygame.init()

        # Create a window
        screen = pygame.display.set_mode((self.width, self.height))
        surface = pygame.surfarray.make_surface(arr)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw the image on the screen
            screen.blit(surface, (0, 0))

            # Update the display
            pygame.display.flip()

        # Quit Pygame
        pygame.quit()

        """
        print(arr)
        img = Image.fromarray(arr)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img)
        ax.axis('off')  # Hide axes

        # Show the figure
        plt.show()
        """
        return True


