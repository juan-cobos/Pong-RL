import numpy as np
import random
import pygame

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
    "black"     : (0, 0, 0),
    "dark_gray"  : (50, 50, 50),
    "mid_gray"  : (128, 128, 128),
    "light_gray": (200, 200, 200),
    "white"     : (255, 255, 255),
}

WIDTH = 256
HEIGHT = 256

class Paddle:
    def __init__(self, x, y, width=10, height=30, speed=10, color="red"):

        #self.shape = width, height
        # TODO: self.pos has to be a 2D with all the positions filled by the Paddle
        # [x: , y: ] * length [0, i] cause it's vertical so j will be the same
        # pos = X, Y coordinates from meshgrid: https://numpy.org/doc/2.2/reference/generated/numpy.meshgrid.html

        self.X, self.Y = np.meshgrid(range(x-width//2, x+width//2), range(y-height//2, y+height//2))
        self._init_X, self._init_Y = self.X, self.Y
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color

    def update(self, action):
        next_Y = self.Y + self.speed * action[1]
        if np.all(0 <= next_Y) and np.all(next_Y < HEIGHT):
            self.Y = self.Y + self.speed * action[1] # Pad only updates in Y axis

    def reset(self):
        self.X, self.Y = self._init_X, self._init_Y

class Ball:
    def __init__(self, x, y, radius, speed=5, color="yellow"):
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

    def update(self):
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
    def __init__(self, paddle_width=None, paddle_height=None, ball_radius=10):
        self.state = np.zeros((WIDTH, HEIGHT, 3), dtype=np.int8)
        self.paddle1 = Paddle(10, HEIGHT//2)
        self.paddle2 = Paddle(WIDTH-10, HEIGHT//2)
        self.ball = Ball(WIDTH//2, HEIGHT//2, radius=ball_radius)

        self.window = None
        self.score = [0, 0]

    def update_state(self):
        arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
        arr[:, :] = COLOR_TO_RGB["black"]  # Init all grid black

        # Fill array with objects at their place
        arr[self.paddle1.X, self.paddle1.Y] =  COLOR_TO_RGB[self.paddle1.color]
        arr[self.paddle2.X, self.paddle2.Y] =  COLOR_TO_RGB[self.paddle2.color]
        arr[self.ball.X, self.ball.Y] =  COLOR_TO_RGB[self.ball.color]

        self.state = arr

    def display_objects(self):
        # Draw ball
        pygame.draw.circle(self.window, self.ball.color, self.ball.center, self.ball.radius)

        # Draw Paddle1
        paddle1_rect = pygame.Rect(self.paddle1.X[0][0], self.paddle1.Y[0][0], self.paddle1.width, self.paddle1.height)
        pygame.draw.rect(self.window, self.paddle1.color, paddle1_rect)

        # Draw Paddle2
        paddle2_rect = pygame.Rect(self.paddle2.X[0][0], self.paddle2.Y[0][0], self.paddle2.width, self.paddle2.height)
        pygame.draw.rect(self.window, self.paddle2.color, paddle2_rect)

    def step(self, action):
        reward = 0
        done = False

        self.paddle1.update(ACTIONS[action])
        self.paddle2.update(ACTIONS[action]) # TODO: Describe policy for Paddle2 (Bot)
        self.ball.update()

        print("Ball array", self.ball.X)
        if np.any(self.ball.X < 0) or np.any(self.ball.X > WIDTH): # It means Goal
            reward = -1 if np.any(self.ball.X < 0) else 1
            self.score += np.array([0, 1] if np.any(self.ball.X < 0) else [1, 0]) # Update score
            self.ball.reset(), self.paddle1.reset(), self.paddle2.reset() # Reset to init positions
            return self.state, reward, True

        if (
                np.any(self.ball.X <= 0) or np.any(self.ball.X >= HEIGHT)
                or np.any(self.ball.Y <= 0) or np.any(self.ball.Y >= HEIGHT)# Outside the grid
                or np.any(np.isin(self.ball.X, self.paddle1.X)) and np.any(np.isin(self.ball.Y, self.paddle1.Y)) # Hits paddle1
                or np.any(np.isin(self.ball.X, self.paddle2.X)) and np.any(np.isin(self.ball.Y, self.paddle2.Y)) # Hits paddle2
        ):
            print("Entra")
            self.ball.dir = -self.ball.dir # Reflect direction

        self.update_state() # Update state with the new object locations
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

        # TODO: Handle resizable window, whether by a scaling factor on the objects or .transform window ish
        font = pygame.font.Font('freesansbold.ttf', 20)  # Use default font, size 36
        clock = pygame.time.Clock()
        background = pygame.Surface((WIDTH, HEIGHT))
        background.fill(COLOR_TO_RGB["black"])

        # Clear screen
        self.window.blit(background, (0, 0))
        # Display objects
        self.display_objects()
        # Render score
        score_text = font.render(f"Score: {self.score[0]} - {self.score[1]}", True, COLOR_TO_RGB["white"])
        self.window.blit(score_text, (20, 20))  # Draw score at top-left

        pygame.display.flip()
        clock.tick(30)



