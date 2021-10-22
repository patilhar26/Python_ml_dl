##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 2
# Description: Using Embedding layer in NN model created previously and check the loss and accuracy results
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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# Embedding that will implement the embedding layer
# Flatten to reshape the arrays
from keras.layers import Embedding, Flatten
# pad_sequences that will be used to pad the sentence sequences to the same length
from keras.preprocessing.sequence import pad_sequences

#Next is importing ‘csv’ file into dataframe.
# We are using pd.read_csv() method of Pandas for that. It creates a Dataframe from a csv file.
# Reading the Data
df = pd.read_csv('imdb_master.csv',encoding='latin-1')
# Printing the head of the data
print(df.head())

# Taking sentences and labels
# .values returns a Numpy array instead of a Pandas Series
df = df[df['label']!='unsup']
sentences = df['review'].values
y = df['label'].values

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

# Creating a Model
model1 = Sequential()
# Adding embedding layers in keras
model1.add(Embedding(vocab_size, 50, input_length=max_review_len))
#  flatten the embedding layer before passing it to the dense layer.
model1.add(Flatten())
# Adding dense layer of 300 units and activation= relu
model1.add(layers.Dense(300, activation='relu',input_dim=max_review_len))
# The neurons in the output layer should be 3 as there are three classes [pos, neg, unsup],
# In the Column label of the dataset,which is the Target.
# Changing the activation function to softmax as it works best for the multi class classification
model1.add(layers.Dense(3, activation='softmax'))
# Compile the model
model1.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
# fit the model
history1=model1.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

# Evaluating the result on test data and get the loss and accuracy values
[test_loss, test_acc] = model1.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


# matplotlib is a plotting library which we have used for our graph
import matplotlib.pyplot as plt
# Plotting history for accuracy
plt.plot(history1.history['acc'])
# Plotting history for val_accuracy
plt.plot(history1.history['val_acc'])
# Plotting History for loss
plt.plot(history1.history['loss'])
# Plotting History for val_loss
plt.plot(history1.history['val_loss'])
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

print('Actual Value is: ',y_test[5],'Predicted Value is: ',model1.predict_classes(X_test[[5],:]))