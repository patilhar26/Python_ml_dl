##-------------------------------------------------------------------------------------------------------------
#  Student Name : Harshita Patil
#  Code for Question 3
# Description: A program to apply the following NLP functions on the "input.txt" and showing the output:
# 1) Tokenization
# 2) POS
# 3) Stemming
# 4) Lemmatization
# 5) Trigram
# 6) Named Entity Recognition
##-------------------------------------------------------------------------------------------------------------

# Imported the essential libraries and created our environment
# The Natural Language Toolkit (NLTK) is a Python package for natural language processing
import nltk

# opening the input.txt in read mode
text = open('input.txt', encoding="utf8").read()

# Tokenization
# word_tokenize() for splitting sentences into word tokens.
wtokens = nltk.word_tokenize(text)
# printing wtokens
print("=========================== Word  Tokenization ==============================", '\n')
print(wtokens)

# sent_tokenize function to tokenize sentences out of paragraph.
stokens = nltk.sent_tokenize(text)
# printing Sentence Tokenization
print("\n========================== Sentence  Tokenization =======================\n")
print(stokens)

# POS
# The POS tagger in the NLTK library outputs specific tags for certain words.
n_pos = nltk.word_tokenize(text)
pos_t = nltk.pos_tag(n_pos)
print("=================================== POS =================================\n")
print("Parts Of Speech: ", pos_t)

# Stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer

# Create instance of PorterStemmer class
ps = PorterStemmer()
# Do stemming on each tokenized word and join them to form sentences.
stemmed_output = ' '.join([ps.stem(w) for w in wtokens])
print("=================================== Stemming =================================", '\n')
print(stemmed_output)

# Create instance of LancasterStemmer class
ls=LancasterStemmer()
# Do stemming on each tokenized word and join them to form sentences.
lsstemmed_output = ' '.join([ls.stem(w) for w in wtokens])
print("=================================== LancasterStemmer =================================", '\n')
print(lsstemmed_output)

# Create instance of SnowballStemmer class
# Choose the language out of all supported - here english
ws=SnowballStemmer("english")
# Do stemming on each tokenized word and join them to form sentences.
wsstemmed_output = ' '.join([ws.stem(w) for w in wtokens])
print("=================================== SnowballStemmer =================================", '\n')
print(wsstemmed_output)

# Lemmatization
from nltk.stem import WordNetLemmatizer
# Create instance of WordNetLemmatizer class
lemmatizer = WordNetLemmatizer()
# Do stemming on each tokenized word and join them to form sentences.
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in wtokens])
print("=================================== Lemmatization =================================", '\n')
print(lemmatized_output)

# Trigram
# Import ngrams from nltk.
from nltk.util import ngrams
# Use n-gram for n = 3
trigrams = ngrams(wtokens,3)
print("====================================== Trigrams ==========================================",'\n')
print(list(trigrams))


# Named Entity Recognition
# Import ne_chunk for getting Named Entity Recognition
from nltk import ne_chunk
#Use ne_chunk on POS tags created above
noe = ne_chunk(pos_t)
print("\nNamed Entity Recognition :", noe)


