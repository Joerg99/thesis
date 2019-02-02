'''
Created on Nov 11, 2018

@author: joerg
'''
import os
import xml.etree.ElementTree as etree
from bs4 import BeautifulSoup
import json

def read_textgrid():
    folder = '/home/joerg/workspace/thesis/Textgrid/'

    for filename in os.listdir(folder):
#         g_id = filename[:8]
        if filename.endswith('.xml'):
            try:
                root = etree.parse(folder+filename).getroot()
            except Exception as e:
                print(filename, e)
        for child in root[1][2][1][0]:
            print(child.tag, child.attrib)


def bs4_textgrid():
#     folder = '/home/joerg/workspace/thesis/gute_Daten/german_tagged/Korpus_Antikoerperchen_Reim_Annotiert/'
    folder = '/home/joerg/workspace/thesis/Textgrid/'
    all_lines = []
    whitespace = "\r\n\t"
    for file in os.listdir(folder):
        if file.endswith('.xml'):
            filename = folder+file
            with open(filename, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file)
                for element in soup.find_all('term'):
                    if element.text == 'verse':
                        for poem in soup.find_all('div', type='text'):
                            try:
                                title = poem.find('head', type = 'h4').text
                            except:
                                title = 'no title'
                            stanza_no = 0
                            for stanza in poem.find_all('lg'):
                                stanza_no  += 1 
                                for line in stanza.find_all('l'):
                                    export_line = '{"s": '+ '"'+line.text.lstrip().strip(whitespace)+'", ' + '"stanza no": '+'"'+str(stanza_no)+'", '+'"title": '+ '"'+title[:15].lstrip().strip(whitespace)+ '"'+'}'
                                    #print(export_line)
                                    all_lines.append(export_line)
                        break

    print(len(all_lines))
    with open("textgrid_l_in_lg_tags_verse_types.txt", 'w') as file:
        for line in all_lines:
            file.write("%s\n" %  line)
                    


def thomas_textgrid_to_ndjson():
    with open('/home/joerg/Downloads/inoutPoetry/textgrid/textgrid.json') as file:
        data = json.load(file)
        for k, v in data.items():
            print(v['year'])


if __name__ == '__main__':
    thomas_textgrid_to_ndjson()
#     read_textgrid()
#     bs4_textgrid()
    
