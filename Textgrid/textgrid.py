'''
Created on Nov 11, 2018

@author: joerg
'''
import os
import re
import xml.etree.ElementTree as etree
from bs4 import BeautifulSoup
import json



def bs4_textgrid():
#     folder = '/home/joerg/workspace/thesis/gute_Daten/german_tagged/Korpus_Antikoerperchen_Reim_Annotiert/'
    folder = '/home/joerg/workspace/thesis/Textgrid/german_untagged/Textgrid/'
    all_lines = []
    whitespace = "\r\n\t"
    poem_no = 0
    for file in os.listdir(folder):
        if file.endswith('.xml'):
            filename = folder+file
            with open(filename, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file)
                try:
                    dates = soup.find_all('date')
                    date = dates[-1]['notbefore']
                except:
                    date='0000'
                for element in soup.find_all('term'): # Dokument nur verarbeiten wenn term == verse
                    if element.text == 'verse':
                        for poem in soup.find_all('div', type='text'): #finde poems in div mit type = text
#                             try:
#                                 title = poem.find('head', type = 'h4').text # name des poems, falls vorhanden
#                             except:
#                                 title = 'no title'
                            stanza_no = 0
                            for stanza in poem.find_all('lg'): # Stanza in poem
                                stanza_alt = stanza_no
                                stanza_no  += 1
                                if stanza_no == 1 and stanza_alt ==0:
                                    poem_no += 1
#                                 print(poem_no, stanza_no)
                                for line in stanza.find_all('l'): # Verse in Stanza
                                    line = re.sub(r'["\'\-\\\/:,\']', '', line.text)
                                    export_line = '{"s": '+ '"'+line.lstrip().strip(whitespace)+'", '+ '"rhyme": ' + '"' + '__'+'", ' + '"poem_no": ' + '"' + str(poem_no)+'", ' + '"stanza_no": '+ '"'+str(stanza_no)+ '", '+ '"released": ' + '"' + str(date)+'"' +'}'

                                    #print(export_line)
                                    all_lines.append(export_line)
                        break

    print(len(all_lines))
    with open("textgrid_l_in_lg_tags_verse_types_re.ndjson", 'w') as file:
        for line in all_lines:
            file.write("%s\n" %  line)
                    
                    
def thomas_textgrid_to_ndjson():
    all_verses = []
    with open('/home/joerg/Downloads/inoutPoetry/textgrid/textgrid.json') as file:
        data = json.load(file)

        for k, v in data.items():
            first_verse = v['lines'][0]
            bad = 0
            for verse in v['lines']:
                if verse == first_verse:
                    bad += 1
                if bad != 2:
                    verse = re.sub(r'["\'\-\\\/:,\']', '', verse)
                    author = v['author'].split(',')[0]
                    export_line = '{"s": '+ '"'+verse+'", '+ '"rhyme": ' + '"' + '__' + '", ' + '"poem_no": ' + '"' + k+'", ' + '"stanza_no": ' + '"' + '__' +'", ' + '"author": '+ '"'+ author + '", '+ '"released": ' + '"' + str(v['year'])+'"' +'}'
                    all_verses.append(export_line)
                else:
                    break
    #print(len(all_verses))
    with open("textgrid_thomas_correct_lines.ndjson", 'w') as file:
        i = 0
        for line in all_verses:
            i+= 1
            file.write("%s\n" %  line)
    print(i)
            
if __name__ == '__main__':
    thomas_textgrid_to_ndjson()
#     bs4_textgrid()
#     text = "Wow was geht."
#     text = re.sub(r'["\'\-\\\/:,\']', '', text)
#     print(text)
