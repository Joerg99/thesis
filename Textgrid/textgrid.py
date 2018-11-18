'''
Created on Nov 11, 2018

@author: joerg
'''
import os
import re
import xml.etree.ElementTree as etree
from bs4 import BeautifulSoup




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
                    
if __name__ == '__main__':
    bs4_textgrid()
#     text = "Wow was geht."
#     text = re.sub(r'["\'\-\\\/:,\']', '', text)
#     print(text)
