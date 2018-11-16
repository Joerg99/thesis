'''
Created on Nov 16, 2018

@author: joerg
'''

# import xml.etree.ElementTree as etree
from bs4 import BeautifulSoup
import os

folder = '/home/joerg/workspace/thesis/Gutenberg/Gutentag/eng_poetry_stanza_only/'
all_lines = []
file_no = 0
for filename in os.listdir(folder):
    file_no += 1
    if filename.endswith('.xml'):
        with open(folder+filename, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file)
            poem_no = 0
            imprint = soup.find('imprint')
            date = imprint.find('date').text
            if len(date) < 4:
                date= '0000'
            
            for element in soup.find_all('lg', type ='poem'): # verses sind in lg
                poem_no += 1
                stanza_no = 0
                for stanza in element.find_all('lg', type='stanza'):
                    if len(stanza) >= 4:
                        stanza_no+=1
                        for verse in stanza.find_all('l'):
                            #print(date, file_no, poem_no, stanza_no, verse.text, filename[:10])
                            all_lines.append([date, file_no, poem_no, stanza_no, verse.text, filename[:10]])


print(len(all_lines))
with open('gutentag_en.txt', 'w') as file:
    for line in all_lines:
        file.write("%s\n" % line)