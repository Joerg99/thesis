'''
Created on Nov 16, 2018

@author: joerg
'''
import os
import re
from bs4 import BeautifulSoup
from collections import Counter



def reader_german_annotated():
    folder = '/home/joerg/workspace/thesis/German_Annotated/german_tagged/Diachron_Sample_DTA_DTR_Rhyme_Annotated/'
    file_no = 0
#     bad_stanza_in_file = Counter()
    all_lines =[]
    line_count = 0
    poem_no = 0
    for filename in os.listdir(folder):
        if filename.endswith('.xml') and not filename.startswith('._'):
            with open(folder+filename, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file)
                file_no += 1

                date = soup.find_all('date', type='publication')[1].text
                if len(date) <4:
                    date = '0000'
                for poem in soup.find_all('lg', type='poem'):
                    poem_no += 1
                    stanza_no = 0
                    rhyme_schema = ''
                    for stanza in soup.find_all('lg', type='stanza'):
                        stanza_no+=1
                        rhyme_schema= stanza['rhyme']
                        stanza_temp = []
                        for verse in stanza.find_all('l'):
#                             verse = re.sub(r"\s+", "", verse.text)
                            verse = verse.text.split()
                            verse_str = ''
                            for token in verse: 
                                verse_str += token+' ' 
                            verse_str = re.sub(r'<[^>]*>', '', verse_str)
                            #verse_str = re.sub(r'["\'-\\\/:,\']', '', verse_str)
#                             print(verse)

                            if len(verse_str) > 1:
                                stanza_temp.append(verse_str)
                        
                            if len(stanza_temp) == len(rhyme_schema):
                                for i in range(len(stanza_temp)):
                                    line_count +=1
                                    export_line = '{"s": '+ '"'+stanza_temp[i].strip()+'", '+ '"rhyme": ' + '"' + rhyme_schema[i]+'", ' + '"poem_no": ' + '"' + str(poem_no)+'", ' + '"stanza_no": '+ '"'+str(stanza_no)+ '", '+ '"released": ' + '"' + str(date)+'"' +'}'
                                    all_lines.append(export_line)
                                    #print(export_line)
    print(line_count)
    with open("dta_annotated.ndjson", 'w') as file:
        for line in all_lines:
            file.write("%s\n" % line)


### Erweiterung um Sonderfälle zu behandeln. Vielleicht benutzen...
# def reader_german_annotated_extrawurst():
#     folder = '/home/joerg/workspace/thesis/gute_Daten/german_tagged/Diachron_Sample_DTA_DTR_Rhyme_Annotated/Extrawurst/'
#     file_no = 0
#     passt_net = 0
#     passt = 0
#     bad_stanza_in_file = Counter()
#     for filename in os.listdir(folder):
#         if filename.endswith('.xml') and not filename.startswith('._'):
#             with open(folder+filename, 'r', encoding='utf-8') as file:
#                 soup = BeautifulSoup(file)
#     
#     #             with open(folder+filename, 'r', encoding='latin-1') as file:
#     #                 for line in file:
#     #                     line = line.strip()
#     #                     line=bytes(line, 'utf-8').decode('ascii','ignore')
#     #                     one_file.append(line)
#     
#                 try:
#                     date = soup.find('date', type='publication').text
#                 except:
#                     date = '0000'
#                     
#                 poem_no = 0
#                 for poem in soup.find_all('lg', type='poem'):
#                     poem_no += 1
#                     for stanza in soup.find_all('lg', type='stanza'):
#                         rhyme_schema= stanza['rhyme']
#                         stanza_temp = []
#                         for verse in stanza.find_all('l'):
#                             #verse = re.sub(r"\s+", "", verse.text)
#                             verse = verse.text.split()
#                             verse_str = ''
#                             for token in verse: 
#                                 verse_str += token+' ' 
#                             verse_str = re.sub(r'<[^>]*>', '', verse_str)
#                             verse = re.sub(r'["\'-\\\/:,\']', '', verse_str)
#                             #print(verse)
# 
#                             if len(verse) > 1:
#                                 if verse[0].isalpha():
#                                     stanza_temp.append(verse)
#                         
#                         if not len(stanza_temp) == len(rhyme_schema):
#                             passt_net += 1
#                             bad_stanza_in_file[filename] += 1
#                             #print(stanza_temp, rhyme_schema, filename)
#                         else:
#                             passt += 1
# #                             print('gut', passt)
#     print(passt, passt_net)
#     print(bad_stanza_in_file)


if __name__ == '__main__':
    reader_german_annotated()