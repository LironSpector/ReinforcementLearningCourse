import pygame, sys, time, random
from pygame.surfarray import array3d
import numpy as np


BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)


class SnakeEnv():
    def __init__(self, frame_size_x, frame_size_y):
        '''
        Defines the initial game window size
        '''
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
        self.reset()

    def reset(self):
        '''
        Resets the game, along with the default snake size and spawning food.
        '''
        self.game_window.fill(BLACK)
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
        self.food_pos = self.spawn_food()
        self.food_spawn = True

        self.direction = "RIGHT"
        self.action = self.direction
        self.score = 0
        self.steps = 0
        print("Game Reset.")

    def change_direction(self, action, direction):
        '''
        Changes direction based on action input.
        Checkes to make sure snake can't go back on itself.
        '''
        if action == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if action == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if action == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if action == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        return direction

    def move(self, direction, snake_pos):
        '''
        Updates snake_pos list to reflect direction change.
        '''
        if direction == 'UP':
            snake_pos[1] -= 10
        if direction == 'DOWN':
            snake_pos[1] += 10
        if direction == 'LEFT':
            snake_pos[0] -= 10
        if direction == 'RIGHT':
            snake_pos[0] += 10

        return snake_pos

    def eat(self):
        '''
        Returns Boolean indicating if Snake has "eaten" the white food square
        '''
        return self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]

    def spawn_food(self):
        '''
        Spawns food in a random location on window size
        '''
        return [random.randrange(1, (self.frame_size_x // 10)) * 10,
                random.randrange(1, (self.frame_size_y // 10)) * 10]

    def human_step(self, event):
        '''
        Takes human keyboard event and then returns it as an action string
        '''
        action = None

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:  # If a key on the keyboard was pressed.
            if event.key == pygame.K_UP:
                action = 'UP'
            if event.key == pygame.K_DOWN:
                action = 'DOWN'
            if event.key == pygame.K_LEFT:
                action = 'LEFT'
            if event.key == pygame.K_RIGHT:
                action = 'RIGHT'
            if event.key == pygame.K_ESCAPE:  # Esc -> Create event to quit the game
                pygame.event.post(pygame.event.Event(pygame.QUIT))

        return action

    def display_score(self, color, font, size):
        '''
        Updates the score in top left
        '''
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x / 10, 15)
        self.game_window.blit(score_surface, score_rect)

    def game_over(self):
        '''
        Checks if the snake has touched the bounding box or itself
        '''
        # TOUCH BOX
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - 10:
            self.end_game()
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - 10:
            self.end_game()

        # TOUCH OWN BODY
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                self.end_game()

    def end_game(self):
        '''

        '''
        message = pygame.font.SysFont('arial', 45)
        message_surface = message.render('Game has Ended.', True, RED)
        message_rect = message_surface.get_rect()
        message_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 4)
        self.game_window.fill(BLACK)
        self.game_window.blit(message_surface, message_rect)
        self.display_score(RED, 'times', 20)
        pygame.display.flip()
        time.sleep(3)
        pygame.quit()
        sys.exit()


snake_env = SnakeEnv(600, 600)

# This is technically a FPS Refresh rate
# Higher number means faster refresh, thus faster snake movement, meaning harder game play
difficulty = 10

# FPS (frames per second) controller
fps_controller = pygame.time.Clock()

# Checks for errors encountered
check_errors = pygame.init()

# Initialise game window
pygame.display.set_caption('Snake Game')

while True:
    # Check Input from Human Step
    for event in pygame.event.get():
        snake_env.action = snake_env.human_step(event)

    # Check for Direction change based on action
    snake_env.direction = snake_env.change_direction(snake_env.action, snake_env.direction)

    # Update Snake Position
    snake_env.snake_pos = snake_env.move(snake_env.direction, snake_env.snake_pos)

    # Check to see if we ate food
    snake_env.snake_body.insert(0, list(snake_env.snake_pos))  # Add a new square to the snake which is going to be the new snake's head (in the X and Y coordinates of the food the snake maybe just ate)
    if snake_env.eat():
        snake_env.score += 1
        snake_env.food_spawn = False
    else:  # If the snake didn't eat food
        snake_env.snake_body.pop()  # Annihilate the 'insert' above

    # Check to see if we need to spawn new food
    if not snake_env.food_spawn:
        snake_env.food_pos = snake_env.spawn_food()
    snake_env.food_spawn = True

    # Draw the Snake
    snake_env.game_window.fill(BLACK)
    for pos in snake_env.snake_body:
        pygame.draw.rect(snake_env.game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))  # (pos[0], pos[1], 10 pixels, 10 pixels)

    # Draw Food
    pygame.draw.rect(snake_env.game_window, WHITE, pygame.Rect(snake_env.food_pos[0], snake_env.food_pos[1], 10, 10))

    # Check if we lost
    snake_env.game_over()

    snake_env.display_score(WHITE, 'consolas', 20)
    # Refresh game screen
    pygame.display.update()

    # Refresh rate
    fps_controller.tick(difficulty)

    img = array3d(snake_env.game_window)  # An image version of the game window
