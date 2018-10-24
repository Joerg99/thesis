'''
Created on 24 Oct 2018

@author: Jorg
'''
import xml.etree.ElementTree as etree

root = etree.parse('test.xml').getroot()

for line in root[1][0]:
    print(line.text)