##-------------------------------------------------------------------------------------------------------------
##  Student Name : Harshita Patil
##  Code for Question 2
## Description:Write a simple program that parse a Wiki page mentioned below and follow the instructions:
#       https://en.wikipedia.org/wiki/Deep_learning
#       Print out the title of the page
#       Find all thelinks in the page (‘a’ tag)
#       Iterate over each tag(above) then return the link using attribute "href" using get
#       Save all the links in the file
##-------------------------------------------------------------------------------------------------------------
#Get the Bautifulsoup library for html parsing and re for string matching
import requests
from bs4 import BeautifulSoup
import re
#storing given url in variable url
url = "https://en.wikipedia.org/wiki/Deep_learning"
Link = requests.get(url)
#parse the html in soup format
Data = BeautifulSoup(Link.content, "html.parser" )
#opening the file in write mode
f = open('link_file.txt','w')
#print title of the page
print("Title of the Page: " + Data.find('title').string)

#using re finding all attributes having either http or https
for link in Data.find_all('a', attrs= {'href': re.compile("^https?://")}):
         print(link.get('href'))
         #writing to the file
         f.write(link.get('href') + '\n')
f.close()
