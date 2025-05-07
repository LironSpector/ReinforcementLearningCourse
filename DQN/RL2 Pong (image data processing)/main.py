from PIL import Image  # To transform the image in the Processor
import numpy as np
import gym
from gym.utils import play

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

env = gym.make("Pong-v0")
num_actions = env.action_space.n


# play.play(env)


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

# Note: if I want to load weights, I should change the value_max=1.0 in the 'LinearAnnealedPolicy' to something like 0.2


# Train the model for 1.5 million steps.
dqn.fit(env, nb_steps=1500000, callbacks=[checkpoint_callback], log_interval=10000, visualize=False)

# After training is done, we save the final weights one more time.
dqn.save_weights(weights_filename, overwrite=True)


# Load my model for evaluation:
# model.load_weights("weights/dqn_BreakoutDeterministic-v4_weights_1200000.h5f")  # Load the weights

# Choose an arbitrary policy for evaluation.
policy = EpsGreedyQPolicy(0.1)

# Initialize the DQNAgent with the new model and updated policy and compile it
dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy, memory=memory,
               processor=processor)
dqn.compile(Adam(lr=0.00025), metrics=['mae'])


dqn.test(env, nb_episodes=5, visualize=False)
# dqn.test(env, nb_episodes=5, visualize=True) - Not working

