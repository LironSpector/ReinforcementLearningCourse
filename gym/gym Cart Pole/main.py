import numpy as np
import matplotlib.pyplot as plt
import gym
import time


NUM_BINS = 10  # The highest 'NUM_BINS' is, better the result would be, and the longer the training time be.

EPOCHS = 20000
ALPHA = 0.8
GAMMA = 0.9

BURN_IN = 1
EPSILON_END = 10000
EPSILON_REDUCE = 0.0001


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


# TRAINING FUNCTIONS AND MORE
def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    '''
    Returns an action for the agent. It uses a random number to decide on
    exploration versus explotation trade-off.
    '''
    random_number = np.random.random()

    # EXPLOITATION, USE BEST Q(s,a) Value
    if random_number > epsilon:

        action = np.argmax(q_table[discrete_state])

    # EXPLORATION, USE A RANDOM ACTION
    else:
        # Return a random 0,1,2,3 action
        action = np.random.randint(0, env.action_space.n)

    return action


def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)


def reduce_epsilon(epsilon, epoch):
    if BURN_IN <= epoch <= EPSILON_END:
        epsilon -= EPSILON_REDUCE
    return epsilon


def fail(done, points, reward):
    if done and points < 150:
        reward -= 200
    return reward


def create_bins(num_bins_per_observation=10):
    bins_cart_position = np.linspace(-4.8, 4.8, num_bins_per_observation)  # bins for the cart position
    bins_cart_velocity = np.linspace(-5, 5, num_bins_per_observation)  # bins for the cart velocity
    bins_pole_angle = np.linspace(-0.418, 0.418, num_bins_per_observation)  # bins for the pole angle
    bins_pole_angular_velocity = np.linspace(-5, 5, num_bins_per_observation)  # bins for the pole angular velocity
    bins = np.array([bins_cart_position, bins_cart_velocity, bins_pole_angle, bins_pole_angular_velocity])  # merge them
    return bins


def discretize_observation(observations, bins, is_initial_state):
    binned_observations = []

    if not is_initial_state:
        observations = (observations,)

    for i, observation in enumerate(observations[0]):
        discretized_observation = np.digitize(observation, bins[i])
        binned_observations.append(discretized_observation)
    return tuple(binned_observations)  # Important for later indexing


env = gym.make("CartPole-v1")
observation = env.reset()  # Reset to initial state
print(f"Observation before changing the values: {observation[0]}")
for step in range(100):
    env.render()  # Render on the screen
    action = env.action_space.sample()  # chose a random action
    observation, reward, done, truncated, info = env.step(action)  # Perform random action on the environment

    if done:
        break

env.close()

print(f"Observation values: {observation}\nReward: {reward}\nDone: {done}\nTruncated: {truncated}\nInfo: {info}")


# Create the bins
bins = create_bins(NUM_BINS)  # Create the bins used for the rest of the code

print(f"Bins for Cart Position: {bins[0]}")
print(f"Bins for Cart Velocity: {bins[1]}")
print(f"Bins for Pole Angle: {bins[2]}")
print(f"Bins for Pole Angular Velocity : {bins[3]}")


# Reset the env and print the different observations.
observations = env.reset()  # env.reset() returns the initial observation
observations = observations[0]
print(f"Cart Position: {observations[0]}")
print(f"Cart Velocity: {observations[1]}")
print(f"Pole Angle: {observations[2]}")
print(f"Pole Angular Velocity : {observations[3]}")

# mapped_observation = discretize_observation(observations, bins)
# print(f"mapped_observation: {mapped_observation}")  # All the values are 5 because I've defined that there are 10 observations, and sinse the all the cart observation values starts close to 0, it belongs to bin 5.


q_table_shape = (NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n) # (NUM_BINS for observation 1, NUM_BINS for obs 2, NUM_BINS for obs 3, NUM_BINS for obs 4, number of possible actions)
q_table = np.zeros(q_table_shape)
print(f"Q table shape: {q_table.shape}\n")  # (10 bins for obs1, 10 bins for obs2, 10 bins for obs3, 10 bins for obs4, 2 actions)


# Exploration vs. Exploitation parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.001  # Exponential decay rate for exploration prob


# VISUALIZATION OF TRAINING PROGRESS PREPARATION
log_interval = 500  # How often do we update the plot? (Just for performance)
render_interval = 10000  # How often to render the game during training (If I want to watch my model learning)

fig = plt.figure()
ax = fig.add_subplot(111)


points_log = []  # to store all achieved points
mean_points_log = []  # to store a running mean of the last 30 results
epochs = []  # store the epochs for plotting
rewards = []  # store the rewards for printing and plotting


start_time = time.time()  # Record the starting time of the entire training process
prev_1000_epochs_time = time.time()  # Record the starting time of the previous 1000 epochs

for epoch in range(EPOCHS):
    loop_st = time.time()

    if epoch % 1000 == 0:
        loop_end = time.time()
        prev_1000_epochs_duration = loop_end - prev_1000_epochs_time
        prev_1000_epochs_time = loop_end  # Update the starting time for the next 1000 epochs

        print(f"Time for the last 1000 epochs: {round(prev_1000_epochs_duration, 2)}s")

    # Continuous State --> Discrete State
    initial_state = env.reset()  # get the initial observation
    discretized_state = discretize_observation(initial_state, bins, is_initial_state=True)  # map the observation to the bins

    done = False  # to stop current run when cartpole falls down
    points = 0  # store result (how many points the cart was able to achieve)
    total_rewards = 0

    # Track Epochs for Plotting Visualization
    epochs.append(epoch)

    while not done:  # Perform current run as long as done is False (as long as the cartpole is up)
        # View how the cartpole is doing every render interval
        if epoch % render_interval == 0:
            env.render()

        action = epsilon_greedy_action_selection(epsilon, q_table, discretized_state)
        next_state, reward, done, truncated, info = env.step(action)  # perform action and get next state
        reward = fail(done, points, reward)  # Subtract from the reward in agent has failed

        next_state_discretized = discretize_observation(next_state, bins, is_initial_state=False)  # map the next observation to the bins

        old_q_value = q_table[discretized_state + (action,)]  # get the old Q-Value from the Q-Table
        next_optimal_q_value = np.max(q_table[next_state_discretized])  # Get the next optimal Q-Value

        next_q_value = compute_next_q_value(old_q_value, reward, next_optimal_q_value)  # Compute next Q-Value
        q_table[discretized_state + (action,)] = next_q_value  # Insert next Q-Value into the table

        discretized_state = next_state_discretized  # Update the old state
        points += 1

        total_rewards = total_rewards + reward


    epsilon = reduce_epsilon(epsilon, epoch)  # Reduce epsilon

    rewards.append(total_rewards)

    points_log.append(points)  # log overall achieved points for the current epoch
    running_mean = round(np.mean(points_log[-30:]), 2)  # Compute running mean points over the last 30 epochs
    mean_points_log.append(running_mean)  # and log it

    if epoch % log_interval == 0:
        print_in_color(f"Total Rewards Overall: {np.sum(rewards)}", color="blue", colors_dict=COLORS_DICT)
        print_in_color(f"Total Rewards from the last 500 runs: {np.sum(rewards[-500:])}", color="green", colors_dict=COLORS_DICT)


    ### Plot the points and running mean ###
    if epoch % log_interval == 0:
        ax.clear()
        ax.scatter(epochs, points_log)
        ax.plot(epochs, points_log)
        ax.plot(epochs, mean_points_log, label=f"Running Mean: {running_mean}")
        plt.legend()
        fig.canvas.draw()

        # Display the plot and add a small delay (0.01 seconds) to allow for smooth updating
        plt.pause(0.01)


env.close()


plt.ticklabel_format(style='plain')
plt.figure(figsize=(10, 6))
plt.plot(range(EPOCHS), np.cumsum(rewards))  # The slope is negative until 10,000 epochs (which is alot) because of my 'fail' function.
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.show()


# Run and display the game after the agent has trained (DISPLAYING NOT WORKING)
observation = env.reset()
observation = observation[0]
rewards = 0
for _ in range(1000):
    env.render()
    discrete_state = discretize_observation(observation, bins, is_initial_state=False)  # get bins
    action = np.argmax(q_table[discrete_state])  # and chose action from the Q-Table
    observation, reward, done, truncated, info = env.step(action)  # Finally perform the action
    rewards += 1
    if done:
        print(f"You got {rewards} points!")
        break

env.close()
