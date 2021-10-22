##-------------------------------------------------------------------------------------------------------------
# Student Name : Harshita Patil
# Code for Question 2
# Description: A program to extract the following web URL text using BeautifulSoup
# https://en.wikipedia.org/wiki/Google
# Saving it in input.txt
##-------------------------------------------------------------------------------------------------------------

# Imported the essential libraries and created our environment
# library to fetch the page content
import requests
# Import the BeautifulSoup class creator from the package bs4.
# bs4(beautiful soup) for parsing the HTML page content.
from bs4 import BeautifulSoup

# Given url
url = 'https://en.wikipedia.org/wiki/Google'
#  res for inspecting the results of the request
res = requests.get(url)
html_page = res.content

# parse the html page
#  The 'html.parser' argument indicates that we want to do the parsing using Python’s built-in HTML parser
soup = BeautifulSoup(html_page, 'html.parser')

# using find method to extract all the div container that have a class attribute of mw-parser-output
data = soup.find('div', {'class': 'mw-parser-output'})

print(data.text)
# opening the input file in write mode
with open('input.txt', 'w', encoding='utf-8') as f:
    f.write(str(data.text))


