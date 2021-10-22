##-------------------------------------------------------------------------------------------------------------
##  Student Name : Harshita Patil
##  Code for Question 3
## Description: A python program to find the wordcount in a file for each line and then print the output and then
# finally store the output back to the file.
##-------------------------------------------------------------------------------------------------------------

# Import the Counter datatype from Collections
from collections import Counter

#openfile in read mode
with open ('Input.txt', 'r') as infile:
#get each record into program variable
    data = infile.read().replace('\n', ' ')

#Remove all the punctuations from input string
    for char in '-.,\n':
        data = data.replace(char, ' ')

#Following translation consists of three things
#    i. Split the input record text into collection of words
#   ii. Use Counter on it to get the unique Words and their number of occurrence
#  iii. Convert the resultant Counter in Dictionary worddict for storing it as Key Value pair
    worddict = dict(Counter(data.split()))

#Open the output file to be written
with open('OutputFilePython.txt', 'w') as outfile:

# Print the contents of dictionary worddict in key value pair
    for key in list(worddict.keys()):
        print(key,":", worddict[key])

# Write the Key Value pair ie. Unique Words in input file and its count
        p = str(key) + " : " + str (worddict[key])
        outfile.write(str(p) + '\n')