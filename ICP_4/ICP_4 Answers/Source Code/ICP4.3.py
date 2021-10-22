##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 3
# Description: A program, to Implement linear Support Vector Machines method using scikit library
# Using the same dataset of previous code
# Using train_test_split to create training and testing part
# Evaluating the model on testing part using score and classification_report(y_true, y_pred)
# We need to show which algorithm we got better accuracy. And to justify why.
##-------------------------------------------------------------------------------------------------------------

# Imported the necessary libraries
# Pandas is a library use for data manipulation and analysis
import pandas as pd

# To implement the Support Vector Machines we will use Scikit-learn and will import our SVM
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# To ignore any deprecation warnings
import warnings
warnings.filterwarnings('ignore')

# Importing the data or reading the data
# read_csv method of Pandas to load the data into a pandas dataframe df
df = pd.read_csv('glass.csv')

# Correct the dataframes by dropping columns
X_train = df.drop("Type",axis=1)
Y_train = df["Type"]

# Using the train_test_split function of scikit-learn to split the data into training and test sets.
# test_size 0.4 indicates we have used 4% of data for testing and random_state ensures reproducibility
X_train, X_test, Y_train, Y_test = train_test_split( X_train, Y_train, test_size = 0.4, random_state=40)

# Create SVM classifier
svc = svm.SVC(kernel="linear")
# Train the model using train sets created above
svc.fit(X_train, Y_train)
# Using predict method on classifier and pass x_test as a parameter to get prediction output
Y_pred1 = svc.predict(X_test)

# Model Accuracy: how often is the classifier correct?
acc_svc = round(accuracy_score(Y_test,Y_pred1)*100,2)

print('\n', '***Accuracy using SVM classifier****' , '\n')
# Print the accuracy score
print("Accuracy score: " + str(acc_svc))
print(classification_report(Y_test,Y_pred1))
