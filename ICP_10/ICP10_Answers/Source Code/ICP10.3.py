##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 3
# Description: Applying the model created previously on 20_newsgroup data set.
##-------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')
# Imported the necessary libraries and created our environment
# Keras is a high-level library.
# Sequential model is a linear stack of layers.
from keras import Sequential
# importing Numpy provides a high-performance multidimensional array and,
# Basic tools to compute with and manipulate these arrays.
import numpy as np
from sklearn import preprocessing
# Importing Tokenizer to tokenize our data
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
# pad_sequences that will be used to pad the sentence sequences to the same length
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
# Embedding that will implement the embedding layer
# Flatten to reshape the arrays
from keras.layers import Flatten
from keras import layers
# fetch_20newsgroups. Specify a download and cache folder for the datasets
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism','sci.space']

# 20newsgroup is standard text classification dataset which is collection of app. 20,000 newsgroups documents.
# Create train dataset from 20newsgroup
newsgroups_train =fetch_20newsgroups(subset='train', shuffle=True,categories=categories)

# Extract data from newsgroups train data set.
sentences = newsgroups_train.data
y = newsgroups_train.target
print(np.unique(y))

# tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
# Preparing the data for embedding layer
max_review_len= max([len(s.split()) for s in sentences])
vocab_size= len(tokenizer.word_index)+1
#getting the vocabulary of data
sentences = tokenizer.texts_to_sequences(sentences)
# padding which indicates whether to add the zeros before or after the sequence.
# # pad_sequences that will be used to pad the sentence sequences to the same length
padded_docs= pad_sequences(sentences,maxlen=max_review_len)
# LabelEncoder to normalize labels.
# Encoding the Target Column
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
# Splitting the data into test and train
# In this test_size = o.25 means 25% od data for testing and 75% of data for training
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)

# create a model
model2 = Sequential()
# Adding embedding layers in keras
model2.add(Embedding(vocab_size, 50, input_length=max_review_len))
#  flatten the embedding layer before passing it to the dense layer.
model2.add(Flatten())
# Adding dense layer of 300 units and activation= relu
model2.add(layers.Dense(300, activation='relu',input_dim=max_review_len))
# Adding dense layer of 20 units and activation= softmax
model2.add(layers.Dense(2, activation='softmax'))
# compile the model
model2.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
# fit the model
historynew=model2.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

# Evaluating the result on test data and get the loss and accuracy values
[test_loss, test_acc] = model2.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# matplotlib is a plotting library which we have used for our graph
import matplotlib.pyplot as plt
# Plotting history for accuracy
plt.plot(historynew.history['acc'])
# Plotting history for val_accuracy
plt.plot(historynew.history['val_acc'])
# Plotting History for loss
plt.plot(historynew.history['loss'])
# Plotting History for val_loss
plt.plot(historynew.history['val_loss'])
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

print('Actual Value is: ',y_test[2],'Predicted Value is: ',model2.predict_classes(X_test[[2],:]))
