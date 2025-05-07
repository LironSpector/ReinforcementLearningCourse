import random
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import deque
import gym
import tensorflow as tf
from keras.models import Sequential, clone_model
from keras.layers import Activation, Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import model_from_json


# # Check if GPU is available and enable memory growth to avoid GPU memory allocation errors
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if physical_devices:
#     try:
#         for device in physical_devices:
#             tf.config.experimental.set_memory_growth(device, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)


ENV_NAME = "CartPole-v1"

# MODEL_PATH = "saved_models"
MODEL_PATH = r"saved_models"


EPOCHS = 1000
BATCH_SIZE = 32  # It's good to choose something in the power of 2 (like 16, 32, 64, 128, 256, 512 and so on), and the larger the hardware, the larger the batch size. The smaller the batch size, the longer the train time takes (because you're feeding in less images at a time.

MIN_EPSILON = 0.01
EPSILON_REDUCE = 0.995  # is multiplied with epsilon each epoch to reduce it
GAMMA = 0.95
LEARNING_RATE = 0.001  # NOT THE SAME AS ALPHA FROM Q-LEARNING FROM BEFORE!!

UPDATE_TARGET_MODEL = 10  # How many epochs is it going to be until I update the target model.

EARLY_STOPPING_THRESHOLD = 150  # threshold for early stopping (number of epochs without improvement)


# Dictionary containing color names and their corresponding ANSI escape codes
COLORS_DICT = {
    "black": ["\033[30m", "\033[0m"],
    "red": ["\033[31m", "\033[0m"],
    "green": ["\033[32m", "\033[0m"],
    "yellow": ["\033[33m", "\033[0m"],
    "blue": ["\033[34m", "\033[0m"],
    "magenta": ["\033[35m", "\033[0m"],
    "cyan": ["\033[36m", "\033[0m"],
    "white": ["\033[37m", "\033[0m"],
    "orange": ["\033[33m", "\033[0m"],
    "purple": ["\033[35m", "\033[0m"],
    "light_gray": ["\033[37m", "\033[0m"],
    "dark_gray": ["\033[90m", "\033[0m"],
    "bright_orange": ["\033[91m", "\033[0m"],
    "pink": ["\033[95m", "\033[0m"],
    "light_blue": ["\033[96m", "\033[0m"],
    "dark_blue": ["\033[34m", "\033[0m"],
    "brown": ["\033[33m", "\033[0m"],
}


def print_in_color(text, color, colors_dict):
    if color in colors_dict:
        begin_color = colors_dict[color][0]
        end_color = colors_dict[color][1]
        print(begin_color + text + end_color)
    else:
        print(text)  # If the specified color is not found, print the text as is


def save_custom_model(model, model_path):
    model_json = model.to_json()
    with open(model_path + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_path + ".h5")


def load_custom_model(model_path):
    with open(model_path + ".json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(model_path + ".h5")
    return model

# FUNCTIONS
def epsilon_greedy_action_selection(model, epsilon, observation):
    if np.random.random() > epsilon:
        observation = np.expand_dims(observation, axis=0)
        prediction = model.predict(observation)  # perform the prediction on the observation
        action = np.argmax(prediction)  # Chose the action with the higher value
    else:
        action = np.random.randint(0, env.action_space.n)  # Else use random action

    return action


def replay(replay_buffer, batch_size, model, target_model):
    # As long as the buffer has not enough elements we do nothing
    if len(replay_buffer) < batch_size:
        return

    # Take a random sample from the buffer with the size of 'batch_size'
    samples = random.sample(replay_buffer, batch_size)
    # print(f"samples:\n{samples}")

    # to store the targets predicted by the target network for training
    target_batch = []

    # Efficient way to handle the sample by using the zip functionality
    zipped_samples = list(zip(*samples))
    # print(f"zipped_samples:\n{zipped_samples}")

    states, actions, rewards, new_states, dones = zipped_samples

    # Predict targets for all states from the sample
    targets = target_model.predict(np.array(states))

    # Predict Q-Values for all new states from the sample
    q_values = model.predict(np.array(new_states))

    # Now we loop over all predicted values to compute the actual targets
    for i in range(batch_size):
        # Take the maximum Q-Value for each sample
        q_value = max(q_values[i][0])

        # Store the ith target in order to update it according to the formula
        target = targets[i].copy()
        if dones[i]:
            target[0][actions[i]] = rewards[i]
        else:
            target[0][actions[i]] = rewards[i] + q_value * GAMMA
        target_batch.append(target)

    # Fit the model based on the states and the updated targets for 1 epoch
    model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)  # setting verbose=0 to not see too much output.


def update_target_model(epoch, update_target_model, model, target_model):
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())


# Create the env and the initial observation
env = gym.make(ENV_NAME)
observation = env.reset()  # Reset to initial state
print(f"Observation before changing the values: {observation[0]}")
for step in range(200):
    env.render()  # Render on the screen
    action = env.action_space.sample()  # chose a random action
    observation, reward, done, truncated, info = env.step(action)  # Perform random action on the environment

    if done:
        break

env.close()

print(f"Observation values: {observation} Reward: {reward} Done: {done} Truncated: {truncated} Info: {info}")


num_actions = env.action_space.n
num_observations = env.observation_space.shape[0]
print(f"There are {num_actions} possible actions and {num_observations} observations")


# Building the model
model = Sequential()

model.add(Dense(16, input_shape=(1, num_observations)))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

# Final Dense Layer
model.add(Dense(num_actions))
model.add(Activation('linear'))  # Has to be linear because essentially we have to choose one of these neurons (the highest value)

model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))  # mse = mean squared error.

# There are 690 params in total: 4 observations * 16(neurons) + 16(bias) + (16 * 32) + 32 + (32 * 2) + 2 = 690
# print(model.summary())

target_model = clone_model(model)


dqn_model_file = os.path.join(MODEL_PATH, "trained_dqn_model")
target_model_file = os.path.join(MODEL_PATH, "trained_target_model")

# Check if the trained models exist
if os.path.exists(dqn_model_file + ".json") and os.path.exists(dqn_model_file + ".h5") \
        and os.path.exists(target_model_file + ".json") and os.path.exists(target_model_file + ".h5"):
    print("Model loaded from the saved_models directory.")
    # Load the trained models
    model = load_custom_model(dqn_model_file)
    target_model = load_custom_model(target_model_file)
else:
    print("Models will be trained and saved to the saved_models directory.\n")

    epsilon = 1.0

    # after the first 20000 values are stored, it needs to remove the oldest value to store the new one.
    replay_buffer = deque(maxlen=20000)  # can contain up to 20000 steps info.

    # Perform the training routine
    best_so_far = 0  # track the highest number of points the agent has earned by keeping the CartPole up.
    points_log = []  # to store all achieved points
    mean_points_log = []  # to store a running mean of the last 30 results
    epochs = []  # store the epochs for plotting
    rewards = []  # store the rewards for printing and plotting

    epochs_without_improvement = 0  # counter for epochs without improvement

    for epoch in range(EPOCHS):  # The training process takes about 5 hours
        observation = env.reset()
        observation = observation[0]

        # Keras expects the input to be of shape [1, X]. Therefore, we have to reshape it.
        num_observations = env.observation_space.shape[0]
        observation = observation.reshape(1, num_observations)

        done = False  # to stop current run when cartpole falls down
        points_agent_gained = 0  # store result (how many points the cart was able to achieve)
        total_rewards = 0

        # Track Epochs for Plotting Visualization
        epochs.append(epoch)

        while not done:

            # Select an action
            action = epsilon_greedy_action_selection(model, epsilon, observation)

            # Perform action and get next state
            next_observation, reward, done, truncated, info = env.step(action)
            next_observation = next_observation.reshape(1, num_observations)

            replay_buffer.append((observation, action, reward, next_observation, done))  # Update the replay buffer
            observation = next_observation  # update the observation
            points_agent_gained += 1  # Works specifically for the cartpole env, because every time the while loop runs (another step has passed) and the game isn't done, 1 is added to the reward.
            total_rewards = total_rewards + reward

            # Most important step! Training the model by replaying
            replay(replay_buffer, BATCH_SIZE, model, target_model)

        epsilon *= EPSILON_REDUCE  # Reduce epsilon
        epsilon = max(epsilon, MIN_EPSILON)

        rewards.append(total_rewards)
        points_log.append(points_agent_gained)  # log overall achieved points for the current epoch.
        running_mean = round(np.mean(points_log[-30:]), 2)  # Compute running mean points over the last 30 epochs
        mean_points_log.append(running_mean)  # and log it

        # Check if we need to update the target model
        update_target_model(epoch, UPDATE_TARGET_MODEL, model, target_model)

        if points_agent_gained > best_so_far:
            best_so_far = points_agent_gained
            epochs_without_improvement = 0  # reset the counter if there's an improvement
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_THRESHOLD:
            print(f"No improvement in the last {EARLY_STOPPING_THRESHOLD} epochs. Stopping training early at epoch {epoch}.")
            break

        last_25_rewards = rewards[-25:]
        if epoch % 25 == 0:
            print_in_color(f"Epoch {epoch}: Points reached: {points_agent_gained} Epsilon: {epsilon} "
                           f"Best so far: {best_so_far}", color="orange", colors_dict=COLORS_DICT)
            print_in_color(f"Avg Reward from the last 25 epochs: {np.mean(last_25_rewards)}", color="red",
                           colors_dict=COLORS_DICT)
            print_in_color(f"Total Rewards Overall: {np.sum(rewards)}", color="blue", colors_dict=COLORS_DICT)
            print_in_color(f"Total Rewards from the last 25 epochs: {np.sum(last_25_rewards)}", color="green",
                           colors_dict=COLORS_DICT)

    # Save the models using the 'save_custom_model' function
    save_custom_model(model, dqn_model_file)
    save_custom_model(target_model, target_model_file)

    # Display a graph of the rewards based on the epochs.
    plt.ticklabel_format(style='plain')
    # plt.figure(figsize=(10, 6))
    print(len(epochs))
    print(len(np.cumsum(rewards)))
    plt.plot(range(epochs), np.cumsum(rewards))  # The slope is negative until 10,000 epochs (which is alot) because of my 'fail' function.
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.show()


# Run and display the game after the agent has trained (DISPLAYING NOT WORKING)
observation = env.reset()
print("O", observation)
observation = observation[0]
num_observations = env.observation_space.shape[0]

rewards = 0
for step in range(400):
    env.render()

    observation = np.expand_dims(observation, axis=0)
    observation = np.expand_dims(observation, axis=1)

    # Get discretized observation
    prediction = model.predict(observation)
    action = np.argmax(prediction)  # Return an array of the actions and their probabilities (e.g [0.4, 0.6] --> We will choose action at index 1)

    observation, reward, done, truncated, info = env.step(action)  # Finally perform the action
    rewards += 1
    print(done, step)
    if done or step == 399:
        print(f"\nYou got {rewards} points!")
        break

env.close()
