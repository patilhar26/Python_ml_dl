##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 2
# Description: A program, to implement Naïve Bayes method using scikit-learn library
# Using glass csv file. Program uses train_test_split to create training and testing part.
# Evaluating the model on testing part using score and classification_report(y_true, y_pred)
##-------------------------------------------------------------------------------------------------------------
#
# Imported the necessary libraries
# Pandas is a library use for data manipulation and analysis
import pandas as pd

# To implement the naive bayes classifier model, we will use scikit-learn
# and will import our GaussianNB from sklearn.naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Importing glass csv input file into dataframe
# We are using read_csv method of Pandas for that
df = pd.read_csv("glass.csv")

# Correct the dataframes by dropping columns
x_train = df.drop("Type",axis=1)
y_train = df["Type"]

# Using the train_test_split function of scikit-learn to split the data into training and test sets.
# test_size 0.4 indicates we have used 4% of data for testing and random_state ensures reproducibility
x_train, x_test, y_train, y_test= train_test_split(x_train, y_train, test_size=0.4, random_state=40)

# Create Gaussian Naive Bayes classifier
classifier = GaussianNB()
# Train the model using train sets created above
classifier.fit(x_train,y_train)

# Using predict method on classifier and pass x_test as a parameter to get prediction output
y_pred = classifier.predict(x_test)

# Model Accuracy: how often is the classifier correct?
accuracy = round(accuracy_score(y_test,y_pred)*100, 2)

print('\n', '***Accuracy using Gaussian Naive Bayes classifier****' , '\n')
# Print the accuracy score
print("Accuracy score: " + str(accuracy))
print(classification_report(y_test, y_pred))

