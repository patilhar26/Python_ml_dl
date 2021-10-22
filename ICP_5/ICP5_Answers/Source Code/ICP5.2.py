##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 2
# Description: A program, to create the Multiple Regression for the “wine quality” dataset.
# In this data set “quality” is the target label.
# Also we have evaluated the model using RMSE and R2 score.
# We have deleted the null values from the data set.
# We have also displayed the top 3 most correlated features to the target label(quality)
##-------------------------------------------------------------------------------------------------------------

# Imported the necessary libraries and created our environment
# Importing pandas library as we will be doing dataframe manipulation,
# and will be using different pandas functions for analysis
import pandas as pd
# Numpy provides a high performance multidimensional array object and tools for working with these array
import numpy as np
# matplotlib is a plotting library for the python programming language and matplotlib.pyplot is a collection of
# command style function which makes it work like MATLAB.
import matplotlib.pyplot as plt

# Importing winequality-red.csv file into dataframe
# We are using read_csv method of Pandas for that
traindf = pd.read_csv('winequality-red.csv')

# select numeric features only
num_features = traindf.select_dtypes(include=[np.number])

# find correlations between all the columns
corr = traindf.corr()
# select top 3 most correlated features with target variable (quality)
print('\n','***Top 3 most correlated features with target variable (quality)***')
print(corr['quality'].sort_values(ascending=False)[:4])
# select top 3 most uncorrelated features with target variable (quality)
print('\n','***Top 3 most uncorrelated features with target variable (quality)***')
print(corr['quality'].sort_values(ascending=False)[-3:])


# Check total number of null values
print('\n',"Total number of null values:  ", + traindf.isnull().sum().sum())

# interpolate all the Null values
traindf = traindf.select_dtypes(include=[np.number]).interpolate().dropna()
print('\n',"Total number of null after interpolation:  ", + traindf.isnull().sum().sum(),'\n')

# separate the features and target values from input dataframe.
# features will be in x and target value in y(quality)
y = np.log(traindf.quality)

# Keep only wanted dataframes by dropping columns
X = traindf.drop(['quality'], axis=1)
#X = traindf.drop(['quality', 'density', 'total sulfur dioxide', 'volatile acidity'], axis=1)


# Using the train_test_split function of scikit-learn to split the data into training and test sets.
# test_size .33 indicates we have used 33% of data for testing and random_state ensures reproducibility.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

# create instance of linear regression model
from sklearn import linear_model
lr = linear_model.LinearRegression()

# fit the model to linear regression instance
model = lr.fit(X_train, y_train)

# calculate r2 score of trained model. Higher the r2 score towards 1 means a better fit model
print ("R^2 is: \n", model.score(X_test, y_test))

# next step is to calculate RMSE. Before that we will use the above built model to get prediction from test data.
predictions = model.predict(X_test)
#print(predictions)

# calculate RMSE score of trained model. Lesser the value means a better fit model.
from sklearn.metrics import mean_squared_error
print("RMSE is: \n", mean_squared_error(y_test, predictions))

# for visualization
actual_values = y_test
# We can view this relationship graphically with scatter plot between predicted and actual values.
# We used color='b' in which b stands for Blue we will get our plot in blue color.
# Alpha helps to show overlapping data
plt.scatter(predictions, actual_values, alpha=.7, color = 'b')
# label for x-axis
plt.xlabel('Predicted Values')
#label for y-axis
plt.ylabel('Actual values')
# set tittle as Linear Regression Model
plt.title('Linear Regression Model')
# To show the Graph
plt.show()