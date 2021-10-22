##-------------------------------------------------------------------------------------------------------------
##  Student Name : Harshita Patil
## Code for Question 2
## Description: This program with given a non-negative integer num, returns the number of steps to reduce it to zero.
# If the current number is even then it will divide it by 2, otherwise, it will subtract 1 from it.
##-------------------------------------------------------------------------------------------------------------

#Get Input from user - Enter a Non negative Integer
var = input("Enter a Non-negative Integer: ")
#convert it into integer and store in variable intv
intv = int(var)

# Initialize counter for number of steps to 0
n = 0

# Perform actions until input number is reduced to 0
while (intv > 0):

#ACTION 1: If input number or previous actions resultant is Even number, then divide by zero
#          Store the resulting value
 if (intv % 2 == 0):
# Hold the original value before taking action for printing
    hld = intv
    intv //= 2
# Once action is performed increment the number of steps
    n += 1
    print("Step " + str(n) +") " + str(hld) + " is even; divide by 2 and obtain: " + str(intv))

#ACTION 2: Else If input number or previous actions resultant is Odd number, then substract 1
#          Store the resulting value
 else:

# Hold the original value before taking action for printing
    hld = intv
    intv -= 1
# Once action is performed increment the number of steps
    n +=1
    print("Step " + str(n) + ") " + str(hld) + " is odd; subtract 1 and obtain: " + str(intv))

# Print the final Output and total number of Steps(n)
print("Output  = " + str(intv))
print("Number of Steps taken to reduce Input to Zero = " + str(n))