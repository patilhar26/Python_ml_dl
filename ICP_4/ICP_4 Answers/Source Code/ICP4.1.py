##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 1
# Description: A program to show the correlation between ‘Survived’ (target column) and and 'Sex’ columns
# for the Titanic use case. Output will help in deciding to keep this feature or not.
##-------------------------------------------------------------------------------------------------------------
#
# Imported the necessary libraries
# Importing pandas library as we will be doing dataframe manipulation,
# and will be using different pandas functions for analysis
import pandas as pd
import matplotlib.pyplot as plt # visualization
import seaborn as sns # statistical visualizations and aesthetics

# Importing train csv file into dataframe
# We are using read_csv method of Pandas for that
train_df = pd.read_csv('./train.csv')
# Importing test csv file. It can be combined with train if required
test_df = pd.read_csv('./test.csv')

# Analysis by pivoting features
print('Correlation between Survived and Pclass')
print(train_df[['Survived','Pclass']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Pclass', ascending=True))

print('\n')
# Below pivoting is done to find correlation between Survived and Sex columns
# GROUPBY on Sex column is used so that mean aggregation function can be used
# on Survived data categorized by Sex values.
print('Correlation between Survived and Sex')
print(train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False))

# For visualizing the correlation we are using Seaborn data visualization library
# We are using Faceting which is act of breaking data variables (here Survived column)
# up across multiple subplots. Afterwards we are combining those subplots into
# single figure.
g = sns.FacetGrid(train_df, col='Survived')

# Calling map function on FacetGrid object built above to graph Sex column.
g.map(plt.hist, 'Sex', bins=20)
plt.show()