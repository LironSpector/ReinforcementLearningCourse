import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# (train_inputs, train_labels), (test_inputs, test_labels) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

# show a single image from the images dataset
plt.imshow(x_train[0])
plt.show()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_categorical_train = to_categorical(y_train, num_classes=10)
y_categorical_test = to_categorical(y_test, num_classes=10)

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation="relu"))
# POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation="relu"))
# POOLING LAYER
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# units = neurons
model.add(Dense(units=256, activation="relu"))

# Choosing 10 neurons for the final Dense layer because this is the output layer and there are 10 different classes
model.add(Dense(units=10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# See some general information about the model I've created
# print(model.summary())

# Early stopping callback
early_stop = EarlyStopping(monitor="val_loss", patience=2)
# Early stop params explanation:
# 'val_loss' means that if there is no improvement in the val_loss after 'patience' epochs, the training will be stopped
# patience: number of epochs with no improvement after which training will be stopped.


model.fit(x_train, y_categorical_train, epochs=15, validation_data=(x_test, y_categorical_test), callbacks=[early_stop])


overall_model_data = pd.DataFrame(model.history.history) # contains data about the loss, accuracy, val_loss and val_accuracy in each epoch
print(f"\noverall_model_data:\n {overall_model_data}")

overall_model_data[["accuracy", "val_accuracy"]].plot()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

overall_model_data[["loss", "val_loss"]].plot()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


print(f"\nModel Evaluation [val_loss, val_accuracy]: {model.evaluate(x_test, y_categorical_test, verbose=0)}")


predictions = model.predict(x_test)
y_pred_categorical = np.argmax(predictions, axis=1)
print(f"\nclassification_report:\n {classification_report(y_true=y_test, y_pred=y_pred_categorical)}")

# Predicting a single image
single_img = x_test[16]
plt.imshow(single_img)

print(f"Real class: {y_test[16]}")  # Printing the real class to compare it with the prediction.
# 1 means single image, (32, 32) means 32 pixels of height and weight, and 3 means 3 color channels (red, green, blue)
single_prediction = model.predict(single_img.reshape(1, 32, 32, 3))
predicted_class = np.argmax(single_prediction)
print(f"predicted_class: {predicted_class}")
