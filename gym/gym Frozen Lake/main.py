import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
import time
from IPython.display import clear_output


# It is common to leave Hyperparameters in ALL CAPS to easily locate them
EPOCHS = 20000  # number of epochs/episodes to train for
ALPHA = 0.8  # aka the learning rate
GAMMA = 0.95  # aka the discount rate
# MAX_EPISODES = 100  # optional, also defined in env setup (register)


# Exploration vs. Exploitation parameters
MAX_EPSILON = 1.0  # Exploration probability at start
MIN_EPSILON = 0.01  # Minimum exploration probability
DECAY_RATE = 0.001  # Exponential decay rate for exploration prob

yoav = 0


# FUNCTIONS FOR
def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    """

    :param epsilon:
    :param q_table:
    :param discrete_state: a number between 0 and 15.
    :return: An action for the agent
    """
    '''
    Returns an action for the agent. Uses a random number to decide on
    exploration versus explotation trade-off.
    '''
    random_number = np.random.random() # number between 0 and 1

    # EXPLOITATION, USE BEST Q(s,a) Value
    if random_number > epsilon:
        # Action row for a particular state
        state_row = q_table[discrete_state, :]
        # Index of highest action for state
        # Recall action is mapped to index (e.g. 0=LEFT, 1=DOWN, etc..)
        action = np.argmax(state_row)

    # EXPLORATION, USE A RANDOM ACTION
    else:
        # Return a random action (0, 1, 2 or 3)
        action = env.action_space.sample()

    return action


def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)


def reduce_epsilon(epsilon, epoch):
    return MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(- DECAY_RATE * epoch)


try:
    register(
        id='FrozenLakeNotSlippery-v0',  # make sure this is a custom name!
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=.8196,  # optimum = .8196
    )
except:
    print('You probably ran this cell twice, accidentally trying to register a new env with the same id twice.')
    print("Either change the id, or just continue, knowing your id was already registered")


env = gym.make("FrozenLakeNotSlippery-v0")  # Load FrozenLake
observation = env.reset()  # Reset to initial state
for step in range(5):
    env.render()  # Render on the screen
    action = env.action_space.sample()  # chose a random action
    observation, reward, terminated, truncated, info = env.step(action)  # Perform random action on the environment

    if terminated:
        env.reset()

env.close()


action_size = env.action_space.n
state_size = env.observation_space.n

# Start with very small values for all our Q(s,a) / 0
q_table = np.zeros([state_size, action_size])

print(f"Q Table at the beginning:\n{q_table}")
print(f"Q Table shape: {q_table.shape}")


# Reset just in case, watch lecture on this.
q_table = np.zeros([state_size, action_size])
epsilon = 1.0  # Exploration rate


# List of rewards
rewards = []
log_interval = 1000
total_reward = 0


# ### VISUALIZATION OF TRAINING PROGRESS ######
# #############################################
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.ion()
# fig.canvas.draw()
# epoch_plot_tracker = []
# total_reward_plot_tracker = []
# ###############################################


# Play 20k games
for episode in range(EPOCHS):
    # Reset the environment
    state = env.reset()[0]
    done = False
    total_rewards = 0

    while not done:
        action = epsilon_greedy_action_selection(epsilon, q_table, state)

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, truncated, info = env.step(action)

        # Look up current/old qtable value Q(s_t,a_t)
        old_q_value = q_table[state, action]

        # Get the next optimal Q-Value Q(st+1, at+1)
        next_optimal_q_value = np.max(q_table[new_state, :])

        # Compute next q value
        next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)
        # print("Next Q")
        # print(next_q)

        # Update Q Table
        q_table[state, action] = next_q
        # print("Q Table")
        # print(q_table)

        total_rewards = total_rewards + reward

        # Our new state is state
        state = new_state

    # Agent finished a round of the game.
    episode += 1

    # Reduce epsilon (because we need less and less exploration)
    epsilon = reduce_epsilon(epsilon, episode)

    rewards.append(total_rewards)

    # total_reward += reward  # Since +1 is only time reward is recorded, meaning game is also done
    # total_reward_plot_tracker.append(total_reward)
    # epoch_plot_tracker.append(episode)

    if episode % log_interval == 0:
        print(f"Total Rewards Overall: {np.sum(rewards)}")
        print(f"Total Rewards from the last 1000 runs: {np.sum(rewards[-1000:])}")

        # ax.clear()
        # ax.plot(epoch_plot_tracker, total_reward_plot_tracker)
        # fig.canvas.draw()
        # plt.show()

env.close()


plt.plot(range(EPOCHS), np.cumsum(rewards))
plt.show()

print(f"\nThe Q Table after all 20,000 epochs:\n{q_table}\n")


# # Reset just in case
# q_table = np.zeros([state_size, action_size])
# epsilon = 1.0  # Exploration rate

# Supposed to show visually how the robot moves after he has trained, BUT IT DOESN'T WORK (IT ISN'T VISUALLY)!
# state = env.reset()[0]
# for _ in range(100):
#     env.render()
#
#     action = np.argmax(q_table[state])  # and chose action from the Q-Table
#     state, reward, done, truncated, info = env.step(action)  # Finally perform the action
#     print(state)
#     print(reward)
#     print(truncated)
#     print(info)
#
#     time.sleep(1)
#     clear_output(wait=True)
#
#     if done:
#         break
#
# env.close()
