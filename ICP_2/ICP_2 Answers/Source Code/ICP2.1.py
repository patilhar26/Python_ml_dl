##-------------------------------------------------------------------------------------------------------------
##  Student Name : Harshita Patil
##  Code for Question 1
## Description: A program, which reads height(feet.) of N students into a list and convert these heights to cm
# in a separate list:
##-------------------------------------------------------------------------------------------------------------

#Get Input from user - Total number of students N
stnum = input("Enter number of students: ")
#convert it into integer and store in variable val
val = int(stnum)

# Define a list to get students height in feet
height_list = []

# Get height of all N number of students and store one by one into the list
for i  in range(val):
    height = input("Enter student " +  str(i)  + " height in feet: ")
    try:
        heightInt = float(height)
        #append all the student height into a list created
        height_list.append(heightInt)

    # catch the exception in case of invalid input from user
    except:
        print("enter and integer")

#Print the list of students height in feet
print("Height in feet: " + str(height_list))

# Define a list to get students height in cm
hght_list_cm = []

for height in height_list:
# Convert all the input heigh of students from feet to cm
    fttocm = height * 30.48
    height_feet = float(format(fttocm, '.1f'))

# Store converted height to list for height in cm
    hght_list_cm.append(height_feet)

# Print the list of students height in cm
print("Height in cm: " + str(hght_list_cm))