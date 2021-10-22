##-------------------------------------------------------------------------------------------------------------
##  Student Name : Harshita Patil
##  Code for Question 2.1


# Ask User to enter input string on Console
# Store the entered string into input Variable
var1 =input("Enter text python: ")
print("Entered text is: ", var1)

# Delete last two character from entered String
var2 = var1.replace(var1[4:6],"")
print("After removing last two chars: ", var2)

# Reverse the resulting String
var3 = var2[::-1]
print("String after reversing the result: ", var3)