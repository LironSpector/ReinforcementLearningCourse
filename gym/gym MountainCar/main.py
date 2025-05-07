# ------ Code to change -------
import numpy as np
import matplotlib.pyplot as plt
import gym
import time


ENV_NAME = "MountainCar-v0"  # Use the exact same name as stated on gym.openai

NUM_BINS = 30  # The highest 'NUM_BINS' is, better the result would be, and the longer the training time be.

EPOCHS = 30000
ALPHA = 0.8
GAMMA = 0.9

BURN_IN = 100
EPSILON_END = 10000
EPSILON_REDUCE = 0.0001

# Early stopping constants
EARLY_STOP_EPISODES = 500  # Number of consecutive episodes to consider for computing the average score


# FUNCTIONS
def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    random_number = np.random.random()

    # EXPLOITATION, USE BEST Q(s,a) Value
    if random_number > epsilon:
        action = np.argmax(q_table[discrete_state])

    # EXPLORATION, USE A RANDOM ACTION
    else:
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


def create_bins(num_bins_per_observation):
    car_position = np.linspace(-1.2, 0.6, num_bins_per_observation)  # bins for the car position
    car_velocity = np.linspace(-0.07, 0.07, num_bins_per_observation)  # bins for the car velocity
    bins = np.array([car_position, car_velocity])  # merge them
    return bins


def discretize_observation(observations, bins, is_initial_state):
    binned_observations = []

    if not is_initial_state:
        observations = (observations,)

    for i, observation in enumerate(observations[0]):
        # print(f"i: {i}")
        # print("obs:", observation)
        discretized_observation = np.digitize(observation, bins[i])
        binned_observations.append(discretized_observation)
    return tuple(binned_observations)  # Important for later indexing


# Create the env and the initial observation
env = gym.make(ENV_NAME)  # use gym.make to create the environment
observations = env.reset()  # Reset to initial state
print(f"Observations before changing the values: {observations[0]}")

observations = observations[0]
print(f"Car Position: {observations[0]}")
print(f"Car Velocity: {observations[1]}")

# Create the bins
bins = create_bins(NUM_BINS)  # Create the bins used for the rest of the code
print(f"Bins for Car Position: {bins[0]}")
print(f"Bins for Car Velocity: {bins[1]}")


# mapped_observation = discretize_observation(observations, bins)
# print(f"mapped_observation: {mapped_observation}")  # All the values are 5 because I've defined that there are 10 observations, and sinse the all the cart observation values starts close to 0, it belongs to bin 5.


q_table_shape = (NUM_BINS, NUM_BINS, env.action_space.n) # (NUM_BINS for observation 1, NUM_BINS for obs 2, number of possible actions)
q_table = np.zeros(q_table_shape)
print(f"Q table shape: {q_table.shape}\n")  # (10 bins for obs1, 10 bins for obs2, 2 actions)


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


max_position_log = []  # to store all achieved points
mean_positions_log = []  # to store a running mean (average while the epochs are still running) of the last 30 results
epochs = []  # store the epochs for plotting
rewards = []  # store the rewards for printing and plotting


best_mean_position = -np.inf  # Initialize the best mean position seen so far
epochs_without_improvement = 0  # Counter for consecutive epochs without improvement


start_time = time.time()  # Record the starting time of the entire training process
prev_1000_epochs_time = time.time()  # Record the starting time of the previous 1000 epochs

for epoch in range(EPOCHS):
    if epoch % 100 == 0:
        print(f"epoch:", epoch)

    loop_st = time.time()

    if epoch % 1000 == 0:
        loop_end = time.time()
        prev_1000_epochs_duration = loop_end - prev_1000_epochs_time
        prev_1000_epochs_time = loop_end  # Update the starting time for the next 1000 epochs

        print(f"Time for the last 1000 epochs: {round(prev_1000_epochs_duration, 2)}s")

    # Continuous State --> Discrete State
    initial_state = env.reset()  # get the initial observation
    discretized_state = discretize_observation(initial_state, bins, is_initial_state=True)  # map the observation to the bins

    done = False
    total_rewards = 0

    max_position = -np.inf  # for plotting

    # Track Epochs for Plotting Visualization
    epochs.append(epoch)

    while not done:  # Perform current run as long as done is False (as long as the cartpole is up)
        # View how the cartpole is doing every render interval
        if epoch % render_interval == 0:
            env.render()

        action = epsilon_greedy_action_selection(epsilon, q_table, discretized_state)
        next_state, reward, done, truncated, info = env.step(action)  # perform action and get next state
        position, velocity = next_state

        next_state_discretized = discretize_observation(next_state, bins, is_initial_state=False)  # map the next observation to the bins

        old_q_value = q_table[discretized_state + (action,)]  # get the old Q-Value from the Q-Table
        next_optimal_q_value = np.max(q_table[next_state_discretized])  # Get the next optimal Q-Value

        next_q_value = compute_next_q_value(old_q_value, reward, next_optimal_q_value)  # Compute next Q-Value
        q_table[discretized_state + (action,)] = next_q_value  # Insert next Q-Value into the table

        discretized_state = next_state_discretized  # Update the old state

        total_rewards = total_rewards + reward

        if position > max_position:  # Only for plotting the results - store the highest point the car is able to reach
            max_position = position

    epsilon = reduce_epsilon(epsilon, epoch)  # Reduce epsilon

    rewards.append(total_rewards)

    max_position_log.append(max_position)  # log the highest position the car was able to reach
    # running_mean = round(np.mean(max_position_log[-30:]), 2)  # Compute running mean points over the last 30 epochs
    running_mean = round(np.mean(max_position_log[-EARLY_STOP_EPISODES:]), 2)  # Compute running mean points over the last 30 epochs
    mean_positions_log.append(running_mean)  # and log it

    # Check for early stopping based on consecutive epochs without improvement
    if running_mean > best_mean_position:
        best_mean_position = running_mean
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOP_EPISODES:
        print(f"No improvement in the last {EARLY_STOP_EPISODES} episodes. Stopping training early at epoch {epoch}.")
        break

    # if epoch % log_interval == 0:
    #     print(f"Total Rewards Overall: {np.sum(rewards)}")
    #     print(f"Total Rewards from the last 1000 runs: {np.sum(rewards[-1000:])}")

    ### Plot the points and running mean ###
    if epoch % log_interval == 0:
        print("bog!", epoch)
        ax.clear()
        ax.scatter(epochs, max_position_log)
        ax.plot(epochs, max_position_log)
        ax.plot(epochs, mean_positions_log, label=f"Running Mean: {running_mean}")
        plt.legend()
        fig.canvas.draw()

        # Display the plot and add a small delay (0.01 seconds) to allow for smooth updating
        plt.pause(0.01)


env.close()


plt.plot(range(EPOCHS), np.cumsum(rewards))
plt.show()


# Run and display the game after the agent has trained (DISPLAYING NOT WORKING)
observation = env.reset()
rewards = 0
for _ in range(1000):
    env.render()
    discrete_state = discretize_observation(observation, bins, is_initial_state=True)  # get bins
    action = np.argmax(q_table[discrete_state])  # and chose action from the Q-Table
    observation, reward, done, truncated, info = env.step(action)  # Finally perform the action
    rewards += 1
    if done:
        print(f"You got {rewards} points!")
        break
env.close()
