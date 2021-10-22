##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 1
# Description: Finding three mistakes in given code
# Explanation of why those mistakes need to be corrected to be able to get the code run.
##-------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')
# Imported the necessary libraries and created our environment
# Keras is a high-level library.
# Sequential model is a linear stack of layers.
from keras.models import Sequential
from keras import layers
# Importing Tokenizer to tokenize our data
from keras.preprocessing.text import Tokenizer
# Importing pandas library as we will be doing dataframe manipulation
import pandas as pd
# importing Numpy provides a high-performance multidimensional array and,
# Basic tools to compute with and manipulate these arrays.
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Next is importing ‘csv’ file into dataframe.
# We are using pd.read_csv() method of Pandas for that. It creates a Dataframe from a csv file.
# Reading the Data
df = pd.read_csv('imdb_master.csv',encoding='latin-1')
# Printing the head of the data
print(df.head())
# Taking sentences and labels
# .values returns a Numpy array instead of a Pandas Series
sentences = df['review'].values
y = df['label'].values

# tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)

# getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)
# Encoding the Target Column
# LabelEncoder to normalize labels.
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Splitting the data into test and train
# In this test_size = o.25 means 25% od data for testing and 75% of data for training
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
# Defining input_dim
# input dimension should be equal to the number of columns of the train dataset after performing train test split,
# which is 2000.
input_dim = np.prod(X_train.shape[1:])
# Number of features
print('input_dim: ',input_dim)

# Creating Model
model = Sequential()
# Adding dense layer of 300 units and activation= relu
model.add(layers.Dense(300,input_dim= input_dim, activation='relu'))
# The neurons in the output layer should be 3 as there are three classes [pos, neg, unsup],
# In the Column label of the dataset,which is the Target.
# Changing the activation function to softmax as it works best for the multi class classification
model.add(layers.Dense(3, activation='softmax'))
# Compile the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
# fit the model
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

# Evaluating the result on test data and get the loss and accuracy values
[test_loss, test_acc] = model.evaluate(X_test,y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# matplotlib is a plotting library which we have used for our graph
import matplotlib.pyplot as plt
# Plotting history for accuracy
plt.plot(history.history['acc'])
# Plotting history for val_accuracy
plt.plot(history.history['val_acc'])
# Plotting History for loss
plt.plot(history.history['loss'])
# Plotting History for val_loss
plt.plot(history.history['val_loss'])
# Printing Title
plt.title('model accuracy')
# y label as loss
plt.ylabel('accuracy')
# x label as epoch
plt.xlabel('epoch')
# Placing a legend on 'accuray', 'validation accuracy','loss','val_loss'
plt.legend(['accuray', 'validation accuracy','loss','val_loss'], loc='upper left')
# To show the graph
plt.show()

print('Actual Value is: ',y_test[4],'Predicted Value is: ',model.predict_classes(X_test[[4],:]))