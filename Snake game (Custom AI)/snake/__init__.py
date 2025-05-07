from gym.envs.registration import register

register(
    id='snake-v0',  # The name to enter when using gym.make()
    entry_point='snake.envs:SnakeEnv',  # Go inside the snake folder --> inside the envs folder --> the name of the class that's going to be inside that environment
)
