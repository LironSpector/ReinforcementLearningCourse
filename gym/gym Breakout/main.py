import gym
from gym.utils import play
import matplotlib.pyplot as plt
import time

env_name = "Breakout-v4"  # Use the exact same name as stated on gym.openai
env = gym.make(env_name, render_mode="rgb_array")  # use gym.make to create your environment

# play.play(env, zoom=2)  # call the play function
env.reset()  # Reset the environment


for _ in range(200):  # Run the environment for 200 steps
    env.render()  # directly displays you the current state of the game

env.close()  # as render always opens a new window when called, it is important to close it again


array = env.render()  # returns the image as a 2d numpy array
print(array)
print(f"Array Shape: {array.shape}")
print(f"Single Pixel Values: {array[60][50]}")  # display the colour of some pixel (row 60 and column 50 - so the red blocks)

plt.imshow(array)  # you can use matplotlib to take a look at your environment - it is the same image as above
plt.show()


print(f"There are {env.action_space.n} possible actions")

for _ in range(300):
    env.render()  # display the current state
    random_action = env.action_space.sample()  # get the random action
    observation, reward, done, truncated, info = env.step(random_action)
    print(f"Reward: {reward}, Done: {done}, Info: {info}")
    # observation, reward, done, info = env.step(random_action) # perform the action the current state of the environment
    # print(f"Reward: {reward}, Done: {done}, Info: {info}")

    if done:
        break

    time.sleep(0.01)  # slow down the game a bit

env.close()  # close the environment

