import time  # to reduce the game speed when playing manually
import gym
from pyglet.window import key  # for manual playing
import pygame

# Import libraries from keras before importing from the rl library
from keras.models import Sequential  # To compose multiple Layers
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam  # Adam optimizer

# Now the keras-rl2 agent.
from rl.agents.dqn import DQNAgent  # Use the basic Deep-Q-Network agent
from rl.memory import SequentialMemory  # Sequential Memory for storing observations (optimized circular buffer)

# LinearAnnealedPolicy allows to decay the epsilon for the epsilon greedy strategy
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


# To see the difference between CartPole-v0 and CartPole-v1:
# https://stackoverflow.com/questions/56904270/difference-between-openai-gym-environments-cartpole-v0-and-cartpole-v1

ENV_NAME = "CartPole-v0"

# Create the env and the initial observation
env = gym.make(ENV_NAME)
observation = env.reset()  # Reset the environment to initial state


rewards = 0
print(f"Observation before changing the values: {observation}\n")
for step in range(200):  # play for max 200 iterations
    env.render()  # render the current game state on your screen
    action = env.action_space.sample()  # chose a random action
    observation, reward, done, info = env.step(action)  # Perform random action on the environment
    rewards += 1

    if done:
        print(f"Agent got {rewards} points before training process\n")
        break

env.close()


print(f"Observation values: {observation} Reward: {reward} Done: {done} Info: {info}\n")

num_actions = env.action_space.n
num_observations = env.observation_space.shape[0]
print(f"There are {num_actions} possible actions and {num_observations} observations")


model = Sequential()

model.add(Flatten(input_shape=(1,) + env.observation_space.shape))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(num_actions))
model.add(Activation('linear'))

print(model.summary())


memory = SequentialMemory(limit=20000, window_length=1)  # Similar to: replay_buffer = deque(maxlen=20000)


policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',  # attribute = epsilon
                              value_max=1.0,  # max epsilon value
                              value_min=0.1,  # min epsilon value
                              value_test=0.05,
                              nb_steps=20000)


dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=100, policy=policy)
# Parameters explanation:
# target_model_update=100 --> Update the target model every 100 epochs


dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])  # mae = mean absolute error


dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)


# After training is done, we save the final weights.
dqn.save_weights(f'dqn_{ENV_NAME}_weights.h5f', overwrite=True)


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
# dqn.test(env, nb_episodes=5, visualize=True) - Not working
env.close()
