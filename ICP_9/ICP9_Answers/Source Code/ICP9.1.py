##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 1
# Description: A program for plotting the Loss and accuracy for both training data and validation data.
# By using the history object in the source code:
# Source Code: https://umkc.box.com/s/10nrlk6216fncengv7qxbbw5o9vgc3hs
##-------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')
# Imported the necessary libraries and created our environment
# Keras is a high-level library.
# Sequential model is a linear stack of layers.
# for mnist digit classification dataset imported mnist
from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

# Loading mnist data set
# This is a dataset of 60,000 28x28 grayscale images of the (0-9) digits, along with a test set of 10,000 images.
# Size of mnist_train_images is 60000 and mnist_test_images is 10000
(train_images,train_labels),(test_images, test_labels) = mnist.load_data()
# printing the images shape.
print('Pixel Values: ', train_images.shape[1:])
# printing the number of train_images
print('Number of train_images and pixel values: ',train_images.shape)
# printing the number of test_images
print('Number of test_images and pixel values: ',test_images.shape)

# Visualizing a image from train_images.
# imshow to display the image
# cmap : This parameter is a colormap instance or registered colormap name used to map scalar data to colors. .
plt.imshow(train_images[0,:,:],cmap='gray')
plt.title('Ground Truth : {}'.format(train_labels[0]))
plt.show()

# Get the number of input neuron(dimension) here 28*28 = 784
dimData = np.prod(train_images.shape[1:])
# Printing dimdata
print('dimdata: ', dimData)
# Convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature.
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

# Making sure that the values are float so that we can get decimal points after division
#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')

# Scale data
# Original data is (0-255). Scale it to range [0,1].
# Normalizing the RGB codes by dividing it to the max RGB value.
train_data /=255.0
test_data /=255.0

#change the labels from integer to one-hot encoding
# one hot encoding is a representation of categorical variables as binary vectors.
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Creating network
# Creating model....
# Sequential is a linear stack of layers.
model = Sequential()
# Adding dense layer of 512 units and activation = relu
# And input dimension is 784 units.
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
# Adding dense layer of 512 unit and activation = relu
model.add(Dense(512, activation='relu'))
# output layer
# 10 output units and activation = softmax
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

# Evaluating the result on test data and get the loss and accuracy values
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Plotting history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
# Printing Title
plt.title('model accuracy')
# y label as accuracy
plt.ylabel('accuracy')
# x label as epoch
plt.xlabel('epoch')
# Placing a legend on 'train' and 'test'
plt.legend(['train', 'test'], loc='upper left')
# To show the graph
plt.show()

# Plotting History for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# Printing Title
plt.title('model loss')
# y label as loss
plt.ylabel('loss')
# x label as epoch
plt.xlabel('epoch')
# Placing a legend on 'train' and 'test'
plt.legend(['train', 'test'], loc='upper left')
# To show the graph
plt.show()


