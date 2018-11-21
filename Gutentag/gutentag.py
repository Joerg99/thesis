'''
Created on Nov 16, 2018

@author: joerg
'''

# import xml.etree.ElementTree as etree
from bs4 import BeautifulSoup
import os
import re

#good for english but only one book for german with 2174 lines

def gutentag_reader():
    folder = '/home/joerg/workspace/thesis/Gutentag/eng_poetry_stanza_only/'
#    folder = '/home/joerg/workspace/thesis/Gutenberg/Gutentag/de_poetry_stanza_only/'
    all_lines = []
    file_no = 0
    poem_no = 0
    for filename in os.listdir(folder):
        file_no += 1
        if filename.endswith('.xml'):
            with open(folder+filename, 'r', encoding='utf-8') as file:
                all_lines=[]
                soup = BeautifulSoup(file)
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
                                verse = re.sub(r'["\'\-\\\/:,\']', '', verse.text)
                                verse = ' '.join(verse.split())
                                export_line = '{"s": '+ '"'+verse.strip()+'", '+ '"rhyme": ' + '"' + '__'+'", ' + '"poem_no": ' + '"' + str(poem_no)+'", ' + '"stanza_no": '+ '"'+str(stanza_no)+ '", '+ '"released": ' + '"' + str(date)+'"' +'}'
                                all_lines.append(export_line)
    
        print(len(all_lines))
        with open('gutentag_en'+str(file_no)+'_'+filename+'.ndjson', 'w') as file:
            for line in all_lines:
                file.write("%s\n" % line)

if __name__ == '__main__':
#     gutentag_reader() 
    folder = '/home/joerg/workspace/thesis/Gutentag/output/'
    all_lines=[]
    i = 0
    for filename in os.listdir(folder):
        i +=1
        if filename.endswith('.ndjson') and i < 2000:
            with open(folder+filename, 'r', encoding='utf-8') as file:
                for line in file:
                    all_lines.append(line)
        print(len(all_lines))
    with open('gutentag_en_all_lines_from_single_file.ndjson', 'w') as file:
        for line in all_lines:
            file.write('%s' % line)
            
