##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 1
# Description: A program to get accuracy and loss based on basic sequential Neural Network Model.
# Then comparing accuracy and loss with adding more dense layer to that model.
##-------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')
# Imported the necessary libraries and created our environment
# Keras is a high-level library.
# Sequential model is a linear stack of layers.
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
# # Importing pandas library as we will be doing dataframe manipulation,
import pandas as pd
import numpy as np
# matplotlib is a plotting library which we will use for our graph
import matplotlib.pyplot as plt


# Next is importing ‘diabetes.csv’ file into dataframe.
# We are using pd.read_csv() method of Pandas for that. It creates a Dataframe from a csv file.
dataset = pd.read_csv("diabetes.csv", header=None).values
# Printing the Shape of dataset
print('Shape of Dataset: ', dataset.shape, '\n')

# Splitting the data into test and train
# In this test_size = o.25 means 25% od data for testing and 75% of data for training
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)

# Using np.random.seed function that sets the random seed of the Numpy
np.random.seed(155)
# creating first Model (first_nn)
# First Model: It has 8 input size and dense layer of 20 units and activation= relu
# Code which was provided in the class
first_nn = Sequential()
# hidden layer
first_nn.add(Dense(20, input_dim=8, activation='relu'))
# output layer
# We took 1 here because it is a binary classification
first_nn.add(Dense(1, activation='sigmoid'))
# Compiling the first model
first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# fitting the model with train data
first_nn_fitted = first_nn.fit(X_train, Y_train, epochs=100,
                               initial_epoch=0)
# Printing the Summary of first Model
print(first_nn.summary())
# Evaluating the first Model
print(first_nn.evaluate(X_test, Y_test))

# Evaluating testing data on first Model
first_evl = first_nn.evaluate(X_test, Y_test)

print('Epoch for first model ends here************************************************************************', '\n')

# Starting 2nd model -------------------------------------------------------------------

# creating second model(second_nn)
# Second Model: In this we are adding dense layers(30, 45, 50) and activation= relu
# To check the changes in accuracy and loss.
second_nn = Sequential()
# hidden layer ( It has 8 input size and dense layer of 20 units)
second_nn.add(Dense(20, input_dim=8, activation='relu'))
# Adding additional dense layer of 30 units
second_nn.add(Dense(30));
# Adding additional dense layer of 45 units
second_nn.add(Dense(45));
# Adding additional dense layer of 50 units
second_nn.add(Dense(50));
# output layer
# We took 1 here because it is a binary classification
second_nn.add(Dense(1, activation='sigmoid'))
# Compiling the second Model
second_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# fitting the data
second_nn_fitted = second_nn.fit(X_train, Y_train, epochs=100,
                               initial_epoch=0)

# Evaluating testing data on Second Model
second_evl = second_nn.evaluate(X_test, Y_test)

print('*************Second model statistics*********************************************************','\n')
# Printing the summary Second Model
print(second_nn.summary())
# Printing the Evaluation of Second Model
print(second_nn.evaluate(X_test, Y_test))

# Compare accuracy and loss of both the models
print('***********************************************************************************************','\n')
print('First models Accuracy: ',first_evl[1],'Second models Accuracy: ',second_evl[1])
print('First models Loss: ',first_evl[0],'Second models Loss: ',second_evl[0])

# Plotting history for accuracy
# first_nn_fitted.history for accuracy
plt.plot(first_nn_fitted.history['acc'])
plt.plot(second_nn_fitted.history['acc'])
# printing title
plt.title("Accuracy on training data")
# y label as accuracy
plt.ylabel('Accuracy')
# x label as epoch
plt.xlabel('Epoch')
# Placing a legend on Model1 and Model2
plt.legend(['Model1', 'Model2'], loc= 'upper left')
# To show the graph
plt.show()

# Plotting History for loss
# first_nn_fitted.history for loss
loss = first_nn_fitted.history['loss']
plt.plot(first_nn_fitted.history['loss'])
plt.plot(second_nn_fitted.history['loss'])
# Printing Title
plt.title('Loss on training data')
# y label as loss
plt.ylabel('Loss')
# x label as epoch
plt.xlabel('Epoch')
# Placing a legend on Model1 and Model2
plt.legend(['Model1', 'Model2'], loc= 'upper left')
# To show the graph
plt.show()