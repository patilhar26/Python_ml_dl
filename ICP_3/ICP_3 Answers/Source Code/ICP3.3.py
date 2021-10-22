##-------------------------------------------------------------------------------------------------------------
##  Student Name : Harshita Patil
##  Code for Question 3
## Description: Write a code Using NumPy create random vector of size 20having only floatin the range 1-20.
#       Then reshape the array to 4by 5
#       Then replace the max in each row by 0(axis=1)
#       (you can NOT implement it via for loop)
##-------------------------------------------------------------------------------------------------------------
#Import the numpy package for multidimensional array hadling under the name np
import numpy as np

#Create a random vector of uniform distribution of Size 2 having float from 1-20
random_vector = np.random.uniform(low=1, high =20, size=20)
print('Initial random vector: \n', random_vector)

#Reshape the above random vector of size 20 to array of dimension (4,5)
n = random_vector.reshape(4,5)
print('\n Reshaped Vector(4,5): \n{}'.format(n))
print('\n Shape(n): ', n.shape)

#Following translation does below things
#    i) Find the max in each row (axis = 1) of random vector n
#   ii) Use numpy where function to replace this maximum element with value 0
# So below translation work as
# n1 = np.where(check if each elements of n (one by one) equals max num on axis 1, yes then value zero, else keep same value)
n1 = np.where(n==np.max(n, axis=1).reshape(-1,1), 0, n)

print('\n Final Vector n1 (replacing Max of n with 0): \n{}'.format(n1))