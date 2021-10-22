##-------------------------------------------------------------------------------------------------------------
##  Student Name : Harshita Patil
##  Code for Question 1
## Description: Create a program of class employee and then do the following:
#   Create a data member to count the number of Employees
#   Create a constructor to initialize name, family, salary, department
#   Create a function to average salary
#   Create a Fulltime Employeeclass and it should inherit the properties of Employee class
#   Create the instance of Fulltime Employee class and Employee class and their member function
##-------------------------------------------------------------------------------------------------------------

#Define Employee class for Employee attributes
class Employee:
    #Initialize all the global variables
    NumOfEmployee = 0
    TotSal = 0
    """Define a constructor for Class instance initialization
    This constructor will get the attributes name, family, department and salary and store them.
    For each instantiation of class Employee global variable 
    NumOfEmployee will be incremented to count number of employee.
    """
    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.NumOfEmployee += 1
        Employee.TotSal += self.salary

   #Define a method to calculate average salary based off all Employee.
    @staticmethod
    def avg_sal():
        Employee.Average = Employee.TotSal / Employee.NumOfEmployee
        return Employee.Average
    # method to print basic info of employee
    def info(self):
        print("Employee:", self.name, self.family,  "Dept:", self.department,"Salary: USD",self.salary)


#Define Subclass FullTimeEmployee inheriting properties of Base class Employee
class FullTimeEmployee(Employee):
    # constructor for this class
    def __init__(self, name, family, salary, department,status ):
         Employee.__init__(self, name, family, salary, department)
         self.status = status
    #define a method to see employee is fulltime or not
    def isfulltime(self):
        if self.status:
            print(self.name, 'is full time employee')
        else:
            print(self.name, 'is not a fulltime employee')

# Create instance of Employee class
e1 = Employee("Harshita","Patil",5000, "IT")
e2 = Employee("Chetan", "Patil",4000, "IT")


#Create instance of FullTimeEmployee class
e3 = FullTimeEmployee("Putul","Patil", 5000, "CS", True)
e4 = FullTimeEmployee("Chiku", "Patil", 7000, "EC", False)

#calling Employee class method
e1.info()
e2.info()
e3.info()
#calling fulltimemployee class method
e3.isfulltime()
#printing average salary and employee count
print("Average Salary: ", Employee.avg_sal())
print("Total Number of Employees: ", Employee.NumOfEmployee)