##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 1
# Description: In this program, we have to change the classifier in the given code provided:
# Source Code: https://umkc.box.com/s/8vygyn9iqj8ut6k8vn434jmpfoldde20
# Changes will be as below:
#   1) Using SVM to check the accuracy.
#   2) Changing the tfidf vectorizer to use bigram and check the accuracy changes. TfidfVectorizer(ngram_range=(1,2))
#   3) Setting argument stop_words='english' to see how accuracy changes.
##-------------------------------------------------------------------------------------------------------------

# Imported the necessary libraries and created our environment
# fetch_20newsgroups. Specify a download and cache folder for the datasets
from sklearn.datasets import fetch_20newsgroups
# The sklearn.feature_extraction module can be used to extract features
from sklearn.feature_extraction.text import TfidfVectorizer
# The sklearn. metrics module implements several loss, score, and utility functions to measure
# classification performance.
from sklearn import metrics
# Get Naive Bayes algorithm
from sklearn.naive_bayes import MultinomialNB
# Support Vector Machines (SVM) and Support Vector Classifier (SVC)
from sklearn.svm import SVC

# 20newsgroup is standard text classification dataset which is collection of app. 20,000 newsgroups documents.
# Create train dataset from 20newsgroup
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# Create TFIDF vector to get weightage of words in input dataset.
tfidf_Vect = TfidfVectorizer()
# Use TFIDF vector transform on train dataset
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)

# Use MultinomialNB to implement Naive Bayes algorithm
clf = MultinomialNB()
# Train model with TFIDF vector from input dataset and train datasets target
clf.fit(X_train_tfidf, twenty_train.target)

# Create test dataset from 20newsgroups
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
# Use TFIDF transform on test dataset
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
# get the predicted values from trained model and test TFIDF vector
predicted = clf.predict(X_test_tfidf)

# calculate accuracy score by comparing predicted vs actual values.
nb_score = round(metrics.accuracy_score(twenty_test.target, predicted) * 100, 2, )

# Change the input code to use SVC classifier instead of NB
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
print('List of all Categories: ')
print(list(twenty_train.target_names), '\n')

# Select fewer categories for faster program performance
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
print('The selected Categories: ')
print(list(twenty_train.target_names),'\n')

# Define method get_score
# This method will :
# 1) Get TFIDF vector transformed dataset (X_train) and the actual TFIDF vector to be used (tfidf_vect).
# 2) Create SVC classifier based model and train it using input TFIDF vector transformed train data
# 3) Create test TFIDF vector transformed data using input TFIDF vector.
# 4) Get the predicted values based of test TFIDF vector dataset
# 5) Calculate accuracy score by comparing predicted vs actual values.
# 6) Return Accuracy Score (score).

def get_score(X_train, tfidf_vect):
    clf = SVC()
    clf.fit(X_train, twenty_train.target)
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
    X_test_tfidf = tfidf_vect.transform(twenty_test.data)
    predicted = clf.predict(X_test_tfidf)
    score = round(metrics.accuracy_score(twenty_test.target, predicted) * 100, 2, )
    return score

# Print accuracy score based off NB algorithm model
print('Accuracy Score (Naive Bayes): ', nb_score,'\n')

# Get train data
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
# Create TFIDF vector to get weightage of words in input dataset.
tfidf_vect1 = TfidfVectorizer()
# Use TFIDF transform on train dataset
X_train_tfidf = tfidf_vect1.fit_transform(twenty_train.data)

# Invoke get_score method to get svc classifier based score
score1 = get_score(X_train_tfidf, tfidf_vect1)
print('Accuracy score (SVM): ', score1, '\n')

# Use bigram TFIDF vector i.e. ngram with n = 2
tfidf_vect2 = TfidfVectorizer(ngram_range=(1, 2))
# Use bigram TFIDF transform on train dataset
X_train_tfidf2 = tfidf_vect2.fit_transform(twenty_train.data)

# Invoke get_score method to get svc classifier and Bigram trained model based score
score2 = get_score(X_train_tfidf2, tfidf_vect2)
print('Accuracy score (Bigram): ', score2, '\n')

# Use stop_words='english' parameter of TFIDF vector.
tfidf_vect3 = TfidfVectorizer(stop_words='english')
# Use stop_words based TFIDF transform on train dataset
X_train_tfidf3 = tfidf_vect3.fit_transform(twenty_train.data)

# Invoke get_score method to get svc classifier and stop_words TFIDF trained model based score
score3 = get_score(X_train_tfidf3, tfidf_vect3)
print('Accuracy score (Stop_words): ', score3, '\n')

