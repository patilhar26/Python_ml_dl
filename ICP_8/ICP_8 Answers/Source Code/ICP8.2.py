##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 2
# Description: A program to change the data source to Breast Cancer dataset and perform the required changes.
# Breast Cancerdataset is designated to predict if a patient has Malignant (M) or Benign = Bcancer
##-------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')
# Imported the necessary libraries and created our environment
# Keras is a high-level library.
# Sequential model is a linear stack of layers.
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
# Importing pandas library as we will be doing dataframe manipulation
import pandas as pd
# matplotlib is a plotting library which we will use for our graph
import matplotlib.pyplot as plt


# Next is importing ‘CC.csv’ file into dataframe.
# We are using pd.read_csv() method of Pandas for that. It creates a Dataframe from a csv file.
# Change the data source to Breast Cancerdataset.
dataset = pd.read_csv('breastcancer.csv')
# Printing 3 samples dataset
print(dataset.sample(3))

# Droping the first and last column fromm dataset
dataset.drop(dataset.columns[[0,32]], axis=1, inplace= True)
print(dataset.sample(3))

# Removing the null values from the dataset
dataset.isnull().sum()

# Separating features and target values
x = dataset.iloc[:, 1:]
y = dataset.iloc[:,0]

# Mapping target column into numerical values
# b= 0(Benign) and M=1(Malignant)
y = y.map({'B':0,'M' :1})
print(y)

# Splitting the data into test and train
# In our dataset 1st column is our target value and other columns are features
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.25, random_state=87)

# Creating Model
first_nn = Sequential()
# hidden layer
# It has 30 input size and dense layer of 40 units and activation= relu
first_nn.add(Dense(40, input_dim=30, activation='relu'))
# output layer
# We took 1 here because it is a binary classification
first_nn.add(Dense(1, activation='sigmoid'))
# Compiling the first Model
first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# Fitting the data
my_first_nn_fitted = first_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)

# Plotting history for accuracy
# first_nn_fitted.history for accuracy
plt.plot(my_first_nn_fitted.history['acc'])
# printing title
plt.title("Accuracy on training data")
# y label as accuracy
plt.ylabel('Accuracy')
# x label as epoch
plt.xlabel('Epoch')
# Placing a legend on Model1 and loc at upper left
plt.legend(['Model1'], loc= 'upper left')
# To show the Graph
plt.show()

# Plotting history for loss
# first_nn_fitted.history for loss
plt.plot(my_first_nn_fitted.history['loss'])

# Printing Title
plt.title('Loss on training data')
# Printing y label
plt.ylabel('Loss')
# Printing X label
plt.xlabel('Epoch')
# Placing a legend on Model1 and loc at upper left
plt.legend(['Model1'], loc= 'upper left')
# To show the Graph
plt.show()

