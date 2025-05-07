import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import load_model


# FILES & PATHS
DATA_DIR = "cell_images"  # dir = directory
TRAIN_PATH = DATA_DIR + "/train"
TEST_PATH = DATA_DIR + "/test"
MODEL_PATH = "saved_models"
MODEL_FILE = "saved_model.pb"
HISTORY_FILE = "history.csv"

# IMAGE CONSTANTS
IMAGE_SHAPE = (130, 130, 3)


# Show the different between an infected and an uninfected cell
uninfected_cell_path = TRAIN_PATH + "/uninfected/" + os.listdir(TRAIN_PATH + "/uninfected")[0]
uninfected_cell = imread(uninfected_cell_path)
plt.imshow(uninfected_cell)
plt.title("Uninfected Cell")
plt.show()

para_cell_path = TRAIN_PATH + "/parasitized/" + os.listdir(TRAIN_PATH + "/parasitized")[0]
parasitized_cell = imread(para_cell_path)
plt.imshow(parasitized_cell)
plt.title("Parasitized Cell")
plt.show()


# Show the number of images in all files
print(f"Num uninfected train images: {len(os.listdir(TRAIN_PATH + '/uninfected'))}")
print(f"Num parasitized train images: {len(os.listdir(TRAIN_PATH + '/parasitized'))}")
print(f"Num uninfected test images: {len(os.listdir(TEST_PATH + '/uninfected'))}")
print(f"Num parasitized test images: {len(os.listdir(TEST_PATH + '/parasitized'))}")

# Find the avg shape of the test images
dim1 = []
dim2 = []
for image_filename in os.listdir(TEST_PATH + "/uninfected"):
    img = imread(TEST_PATH + "/uninfected/" + image_filename)
    d1, d2, colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

# Create a DataFrame with dim1 and dim2 as columns
data = pd.DataFrame({'dim1': dim1, 'dim2': dim2})

# Use jointplot with the DataFrame
sns.jointplot(data=data, x='dim1', y='dim2')
plt.show()

print(f"Avg value in dim 1: {round(np.mean(dim1), 2)}\nAvg value in dim 2: {np.mean(dim2)}")


image_gen = ImageDataGenerator(rotation_range=20,  # rotate the image 20 degrees
                               width_shift_range=0.10,  # Shift the pic width by a max of 5%
                               height_shift_range=0.10,  # Shift the pic height by a max of 5%
                               rescale=1/255,  # Rescale the image by normalzing it.
                               shear_range=0.1,  # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1,  # Zoom in by 10% max
                               horizontal_flip=True,  # Allo horizontal flipping
                               fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                            )

print(image_gen.flow_from_directory(TRAIN_PATH))
print(image_gen.flow_from_directory(TEST_PATH))


# Creating the model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=IMAGE_SHAPE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=IMAGE_SHAPE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=IMAGE_SHAPE, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, its binary so we use sigmoid
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# print(model.summary())


early_stop = EarlyStopping(monitor='val_loss',patience=2)

batch_size = 8  # It's good to choose something in the power of 2 (like 16, 32, 64, 128, 256, 512 and so on), and the larger the hardware, the larger the batch size. The smaller the batch size, the longer the train time takes (because you're feeding in less images at a time.


train_image_gen = image_gen.flow_from_directory(TRAIN_PATH,
                                                target_size=IMAGE_SHAPE[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(TEST_PATH,
                                                target_size=IMAGE_SHAPE[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary',
                                               shuffle=False)

# See which class is 0 and which is 1:
print(train_image_gen.class_indices)


# TRAIN / LOAD the model
model_exists = os.path.exists(os.path.join(MODEL_PATH, MODEL_FILE))
history_exists = os.path.exists(os.path.join(MODEL_PATH, HISTORY_FILE))


if model_exists:
    model = load_model(MODEL_PATH)
    print("Model loaded from the saved_models directory.")
    if history_exists:
        overall_model_data = pd.read_csv(os.path.join(MODEL_PATH, HISTORY_FILE))
        print("History loaded from the saved_models directory.")
    else:
        overall_model_data = pd.DataFrame()
        print("History file not found. Starting a new history.")
else:
    results = model.fit_generator(train_image_gen,
                                  epochs=20,
                                  validation_data=test_image_gen,
                                  callbacks=[early_stop])

    model.save(filepath=MODEL_PATH)

    overall_model_data = pd.DataFrame(model.history.history)
    overall_model_data.to_csv(os.path.join(MODEL_PATH, HISTORY_FILE), index=False)
    print("Model trained. Both model and history saved to the saved_models directory.")


# overall_model_data contains data about the loss, accuracy, val_loss and val_accuracy in each epoch.
if not overall_model_data.empty:
    print(f"\noverall_model_data:\n {overall_model_data}")

    overall_model_data[["accuracy", "val_accuracy"]].plot()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    overall_model_data[["loss", "val_loss"]].plot()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


print(f"\nModel Evaluation [val_loss, val_accuracy]: {model.evaluate_generator(test_image_gen)}")

pred_probabilities = model.predict_generator(test_image_gen) # the values in this array aren't the classification like 0 or 1, but the probability the computer thinks they belong to a certain class.
print(f"Predictions:\n{pred_probabilities}")
predictions = pred_probabilities > 0.5  # if the model is more than 50% sure in his prediction, than give it to one class, otherwise, give it to the second class.
print(f"Predictions:\n{predictions}")

print(f"pred_probabilities length: {len(pred_probabilities)}")

print(f"y_test values: {test_image_gen.classes}")  # An array that contains the actual classes of each picture (y_test values)

print(f"\nclassification_report:\n {classification_report(y_true=test_image_gen.classes, y_pred=predictions)}")
print(f"confusion_matrix:\n {confusion_matrix(y_true=test_image_gen.classes, y_pred=predictions)}\n")


# Predicting a single/new image
para_cell_path = TRAIN_PATH + "/parasitized/" + os.listdir(TRAIN_PATH + "/parasitized")[0]
my_image = Image.open(para_cell_path).resize(IMAGE_SHAPE[:2])  # Open the image and resize it to match the model's input shape

my_img_arr = np.array(my_image)  # Convert the PIL Image to a NumPy array

print(f"Shape before changing {my_img_arr.shape}")
# We need to adapt the image shape for the model
my_img_arr = np.expand_dims(my_img_arr, axis=0)
print(f"Shape after changing {my_img_arr.shape}")

single_prediction = model.predict(my_img_arr)
print(f"single_prediction: {single_prediction}")

# See which class is 0 and which is 1:
print(train_image_gen.class_indices)  # parasitized: 0, uninfected: 1.
if single_prediction == 0:
    print("The cell is parasitized")
else:
    print("The cell is uninfected")
