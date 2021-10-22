##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 1
# Description: A program, to built scatter plot between GarageArea and SalePrice.
# And to delete all the outlier data for the GarageArea field
# (for the same data set in the use case: House Prices).
##-------------------------------------------------------------------------------------------------------------

# Imported the necessary libraries and created our environment
# Importing pandas library as we will be doing dataframe manipulation,
import pandas as pd
# matplotlib is a plotting library for the python programming language and matplotlib.pyplot is a collection of
# command style function which makes it work like MATLAB.
import matplotlib.pyplot as plt

# Importing train.csv file into dataframe
# We are using read_csv method of Pandas for that
traindf = pd.read_csv('./train.csv')

# By using mptools we can choose style . In this we used "ggplot" style, which adjust the style to emulate ggplot.
plt.style.use(style='ggplot')
# figsize  is width, height in inches
plt.rcParams['figure.figsize'] = (10, 6)

# we will set the label for the x-axis and y axis
plt.xlabel('Garage area')
plt.ylabel('Sales Price')


# We can view this relationship graphically with scatter plot between GarageArea and SalePrice.
# First we will create a graph of GarageArea and SalePrice without filtering any data
# We used color='b' in which b stands for Blue we will get our plot in blue color
unfilter_data = traindf[['GarageArea','SalePrice']]
plt.scatter(unfilter_data['GarageArea'],unfilter_data['SalePrice'],color='b')
# set tittle as Unfiltered Graph
plt.title('Unfiltered Graph')
# to show the graph
plt.show()

# Based on above graph we will remove the outlier data for the GarageArea field to make it more linear
filter_data = traindf[(traindf['GarageArea'] < 1000) & (traindf['SalePrice'] < 450000) & (traindf['GarageArea'] > 100)]
plt.scatter(filter_data['GarageArea'],filter_data['SalePrice'],color='b')

plt.xlabel('Garage area')
plt.ylabel('Sales Price')

# set tittle as Filtered Graph
plt.title('Filtered Graph')
# to show the graph
plt.show()