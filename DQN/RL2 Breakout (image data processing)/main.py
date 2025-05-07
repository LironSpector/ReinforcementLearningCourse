# from PIL import Image  # To handle images
# import matplotlib.pyplot as plt  # For plotting
# import numpy as np
# import gym
# # from gym.utils import play  # to play manually
#
#
# import tensorflow as tf
# from tensorflow import keras
# from rl.core import Processor  # To process the image within the keras-rl training routine
# from rl.memory import SequentialMemory  # To store the sequential frames
#
# from collections import deque
#
#
# WINDOW_LENGTH = 3  # Three frames in a row of the game in each window
#
# env = gym.make("Breakout-v0")
#
# # play.play(env)
#
#
# np.random.seed(42)
# env.reset()
#
# # sequential_frame_buffer example (e = experience): [e0, e1, e2] or in more details: [[0, 1, 2], [1, 2, 3], [2, 3, 4]] - each number is a frame inside a window.
# sequential_frame_buffer = []  # Our actual memory
#
# # Temporary storage to capture sequential frames which can store a max of WINDOW_LENGTH images
# temp_sequential_frames = deque(maxlen=WINDOW_LENGTH)
#
#
# for i in range(10):
#     if i == 1:
#         action = 1  # Initiate ball
#     else:
#
#         action = 3  # always go left, to visualize the movement
#     observation, reward, done, info = env.step(action)  # and perform it on the environment to get the next state
#
#     # We have to wait until the deque is full (so it contains exactly WINDOW_LENGTH images)
#     if len(temp_sequential_frames) == WINDOW_LENGTH:
#         print(i)
#         # If the deque is full we know that it contains WINDOW_LENGTH frames and we append those frames to our actual memory
#         sequential_frame_buffer.append(list(temp_sequential_frames))
#
#     # Update the deque
#     temp_sequential_frames.append(observation)
#
# print(sequential_frame_buffer)
# print(len(sequential_frame_buffer))  # Should be 7 because: 10 - WINDOW_LENGTH = 3
# print(len(sequential_frame_buffer[0]))  # Should be 3 because WINDOW_LENGTH = 3
#
# plt.imshow(sequential_frame_buffer[0][0])
# plt.show()
#
#
# fig, axis = plt.subplots(4, WINDOW_LENGTH, figsize=(12, 12))
#
# for global_index, timestep in enumerate(sequential_frame_buffer[:4]):
#     for frame_index, frame in enumerate(timestep):
#         axis[global_index][frame_index].imshow(frame)
#
# fig.subplots_adjust(wspace=-0.7, hspace=0.5)
# plt.show()
#
#
# # does the same thing in a much more optimized way
# memory = SequentialMemory(limit=1000, window_length=WINDOW_LENGTH)
#
# # Besides that we need to decide how large our images should be.
# # Larger images might contain more information but also increase the training time.
# # Let us use an image size of  84×84
# IMG_SHAPE = (84, 84)
#
#
# class BreakOutProcessor(Processor):
#     def process_observation(self, observation):
#         # First convert the numpy array to a PIL Image
#         img = Image.fromarray(observation)
#         # Then resize the image
#         img = img.resize(IMG_SHAPE)
#         # And convert it to grayscale
#         img = img.convert("L")
#         # Finally we convert the image back to a numpy array and return it
#         return np.array(img)
#
#
# # Try the processor
# sample_images = []
# breakout_processor = BreakOutProcessor()
# env.reset()
# for _ in range(200):
#     action = env.action_space.sample()  # sample a random action
#     observation, reward, done, info = env.step(action)  # and perform it on the environment to get the next state
#     processed_observation = breakout_processor.process_observation(observation)
#     sample_images.append(processed_observation)
#
# print(f"The shape of the original observation is {observation.shape} "
#       f"and the shape of the processed observation is {processed_observation.shape}")
#
#
# plt.figure()
# plt.imshow(sample_images[-1], cmap="gray")  # Display a single img after resizing it and turning it into a grayscale img
# plt.show()



from PIL import Image  # To transform the image in the Processor
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


IMG_SHAPE = (84, 84)
WINDOW_LENGTH = 4

env = gym.make("BreakoutDeterministic-v4")
# env = gym.make("BreakoutDeterministic-v4", render_mode='human')
num_actions = env.action_space.n


class ImageProcessor(Processor):
    def process_observation(self, observation):
        # First convert the numpy array to a PIL Image
        img = Image.fromarray(observation[0])
        # Then resize the image
        img = img.resize(IMG_SHAPE)
        # And convert it to grayscale  (The L stands for luminance)
        img = img.convert("L")
        # Convert the image back to a numpy array and finally return the image
        img = np.array(img)
        return img.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We divide the observations by 255 to compress it into the intervall [0, 1].
        # This supports the training of the network
        # We perform this operation here to save memory.
        processed_batch = batch.astype('float32') / 255.0
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)


input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])
print(f"Input shape: {input_shape}")


model = Sequential()

# Permute to the shape the Convolution2D wants: (None, 84, 84, 4) or (Batch, 84, 84, 4)
model.add(Permute((2, 3, 1), input_shape=input_shape))

model.add(Convolution2D(filters=32, kernel_size=(8, 8), strides=(4, 4), kernel_initializer='he_normal'))
model.add(Activation('relu'))

model.add(Convolution2D(64, (4, 4), strides=(2, 2), kernel_initializer='he_normal'))
model.add(Activation('relu'))

model.add(Convolution2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(num_actions))
model.add(Activation('linear'))

print(model.summary())


memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)  # A replay buffer

processor = ImageProcessor()

# We use again a LinearAnnealedPolicy to implement the epsilon greedy action selection with decaying epsilon.
# As we need to train for at least a million steps, we set the number of steps to 1,000,000
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',  # attribute = epsilon
                              value_max=1.0,  # max epsilon value
                              value_min=0.1,  # min epsilon value
                              value_test=0.05,  # An epsilon value for testing the model. It's good to choose a very low probability to choose a random action.
                              nb_steps=1000000)


dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=0.99, target_model_update=10000,
               train_interval=4, delta_clip=1)
# Parameters explanation:
# target_model_update=1000 --> Update the target model every 1000 epochs - should be in line to the same factor of the 'nb_steps' in the 'LinearAnnealedPolicy' as well as the 'limit' in the 'SequentialMemory'
# train_interval=4 --> train the model every 4th step because I'm taking those 4 frames at a time.


dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])  # mae = mean absolute error


# As the training might take several hours, we store our current model each 100,000 steps.
# We can use the ModelIntervalCheckpoint(checkpoint_name, interval) to do so and store it in a callback variable which we pass to the fit method as a callback
weights_filename = 'dqn_breakout_weights.h5f'  # Our final weights
checkpoint_weights_filename = 'dqn_' + "BreakoutDeterministic-v4" + '_weights_{step}.h5f'
checkpoint_callback = ModelIntervalCheckpoint(checkpoint_weights_filename, interval=100000)


# Train the model for 1.5 million steps.
# dqn.fit(env, nb_steps=1500000, callbacks=[checkpoint_callback], log_interval=10000, visualize=False)

# After training is done, we save the final weights one more time.
# dqn.save_weights(weights_filename, overwrite=True)


# Note: if I want to load weights, I should change the value_max=1.0 in the 'LinearAnnealedPolicy' to something like 0.2

# -- Load my model for evaluation: --
model.load_weights("weights/dqn_BreakoutDeterministic-v4_weights_1200000.h5f")  # Load the weights

# Choose an arbitrary policy for evaluation.
policy = EpsGreedyQPolicy(0.1)

# Initialize the DQNAgent with the new model and updated policy and compile it
dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy, memory=memory,
               processor=processor)
dqn.compile(Adam(lr=0.00025), metrics=['mae'])


# Evaluating the model
dqn.test(env, nb_episodes=5, visualize=False)
# dqn.test(env, nb_episodes=5, visualize=True) - Not working

