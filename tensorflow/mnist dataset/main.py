import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Visualizing the image data
single_image = x_train[0]
print(f"Single image:\n{single_image}")
print(f"Single image shape: {single_image.shape}\n")
plt.imshow(single_image)
plt.show()

"""We first need to make sure the labels will be understandable by our CNN."""
"""Hmmm, looks like our labels are literally categories of numbers. We need to translate this to be "one hot encoded" so our CNN can understand, otherwise it will think this is some sort of regression problem on a continuous axis. Luckily , Keras has an easy to use function for this:"""
y_cat_train = to_categorical(y_train, num_classes=10)
y_cat_test = to_categorical(y_test, num_classes=10)

# normalize the X data
x_train = x_train / 255
x_test = x_test / 255

scaled_single = x_train[0]

plt.imshow(scaled_single)
plt.show()

"""Reshaping the data:
right now our data is 60,000 images stored in 28 by 28 pixel array formation.
This is correct for a CNN, but we need to add one more dimension to show we're dealing with 1 RGB channel (since technically the images are in black and white, only showing values from 0-255 on a single channel), an color image would have 3 dimensions.
"""
# Reshape to include channel dimension (in this case, 1 channel)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


"""Training the Model"""
model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu'))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, SO THERE ARE 10 CLASSES
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# See some general information about the model I've created
# print(model.summary())


early_stop = EarlyStopping(monitor='val_loss', patience=2)
# Early stop params explanation:
# 'val_loss' means that if there is no improvement in the val_loss after 'patience' epochs, the training will be stopped
# patience: number of epochs with no improvement after which training will be stopped.

"""Train the Model"""
model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_stop])

"""Evaluate the Model"""
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


print(f"\nModel Evaluation [val_loss, val_accuracy]: {model.evaluate(x_test, y_cat_test, verbose=0)}")


predictions = model.predict(x_test)
y_pred_categorical = np.argmax(predictions, axis=1)
print(f"\nclassification_report:\n {classification_report(y_true=y_test, y_pred=y_pred_categorical)}")
print(f"confusion_matrix:\n{confusion_matrix(y_test, y_pred_categorical)}")


# Predicting a single number image
my_number = x_test[0]
plt.imshow(my_number.reshape(28, 28))
plt.show()

single_prediction = model.predict(my_number.reshape(1, 28, 28, 1))  # SHAPE --> (num_images,width,height,color_channels)
predicted_class = np.argmax(single_prediction)
print(f"predicted_class: {predicted_class}")
