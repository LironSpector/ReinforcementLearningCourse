import gym
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


ENV_NAME = "Acrobot-v1"

# Create the env and the initial observation
env = gym.make(ENV_NAME)
# env = ObservationWrapper(gym.make(ENV_NAME))
observation = env.reset()  # Reset the environment to initial state


num_actions = env.action_space.n
num_observations = env.observation_space.shape[0]
print(f"There are {num_actions} possible actions and {num_observations} observations")


print(f"Observation before changing the values: {observation}\n")
for step in range(200):  # play for max 200 iterations
    env.render()  # render the current game state on your screen
    action = env.action_space.sample()  # chose a random action
    observation, reward, done, info = env.step(action)  # Perform random action on the environment

    if done or step == 199:
        # print(f"Agent got {reward} points before training process\n")
        break

env.close()


print(f"Observation values: {observation} Reward: {reward} Done: {done} Info: {info}\n")


model = Sequential()


model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Flatten(input_shape=(1, num_observations)))
# model.add(Flatten(input_shape=(num_observations,)))


model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))


model.add(Dense(num_actions))
model.add(Activation('linear'))

# print(model.summary())


memory = SequentialMemory(limit=50000, window_length=1)  # Similar to: replay_buffer = deque(maxlen=50000)


policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',  # attribute = epsilon
                              value_max=1.0,  # max epsilon value
                              value_min=0.1,  # min epsilon value
                              value_test=0.05,  # An epsilon value for testing the model. It's good to choose a very low probability to choose a random action.
                              nb_steps=150000)

# If I want to load weights, I should change the value_max=1.0 in the 'LinearAnnealedPolicy' to something like 0.2
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
#                               attr='eps',  # attribute = epsilon
#                               value_max=0.2,  # max epsilon value
#                               value_min=0.1,  # min epsilon value
#                               value_test=0.05,  # An epsilon value for testing the model. It's good to choose a very low probability to choose a random action.
#                               nb_steps=150000)


dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1000, policy=policy, batch_size=32, gamma=0.99)
# Parameters explanation:
# target_model_update=1000 --> Update the target model every 1000 epochs


dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])  # mae = mean absolute error


dqn.fit(env, nb_steps=150000, visualize=False, verbose=2)


# After training is done, we save the final weights.
dqn.save_weights(f'dqn_{ENV_NAME}_weights.h5f', overwrite=True)


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
# dqn.test(env, nb_episodes=5, visualize=True) - Not working
env.close()
