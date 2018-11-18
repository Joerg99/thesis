'''
Created on Nov 10, 2018

@author: joerg
'''
import ndjson
import xml.etree.ElementTree as etree
import os
import re
from bs4 import BeautifulSoup

# 37535 Zeilen
def xml_to_string():
    # beide Ordner benutzen!!!!!
    # __kaputt_linux verursacht Fehler beim einlesen, Zeilen werden aber trotzdem erkannt
    #folder = '/home/joerg/workspace/thesis/gute_Daten/german_tagged/Diachron_Sample_DTA_DTR_Rhyme_Annotated/'
    folder = '/home/joerg/workspace/thesis/gute_Daten/german_tagged/Diachron_Sample_DTA_DTR_Rhyme_Annotated/__kaputt_linux'


    poem = []
    broken = 0
    for filename in os.listdir(folder):
        g_id = filename[:8]
        if filename.endswith('.xml'):
            try:
                root = etree.parse(folder+filename).getroot()
            except Exception as e:
#                 print('Fileread error: ', e)
#                 print('Could not read:',  filename)
                continue#break 
            for child in root[1].iter():
                if child.tag == '{http://www.tei-c.org/ns/1.0}lg':
                    if 'rhyme' in child.attrib:
                        try:
                            stanza_number = child.attrib['n']
                        except:
                            stanza_number = 0
                        #print(child.attrib['rhyme'], len(child.attrib['rhyme']))
                        stanza_length = len(child.attrib['rhyme'])
                          
                        stanza = child
                           
                        i = 0
                        v = []
                        for verse in stanza.iter():
                            if verse.tag == '{http://www.tei-c.org/ns/1.0}l':
                                v.append(verse.text)
                                i +=1
                        if i == stanza_length: # check if all lines within a stanza are captured
                            #print(child.attrib['rhyme'])
                            for j in range(stanza_length):
                                try:
                                    poem.append('{"s": '+ '"'+v[j]+'", '+ '"rhyme": '+ '"'+child.attrib['rhyme'][j]+'", '+'"stanza": '+ '"'+str(stanza_number)+'", '+ '"g_id": '+ '"'+str(g_id)+ '"'+'}')
                                except Exception as e:
                                    broken +=1
#                                     print('Error creating line',e)
#                                     print(v[j], child.attrib['rhyme'][j], str(g_id))
                                    continue
                                 
    print('kaputte Zeilen: ', broken)
    with open("dta_gold_alle_etree_linuxtest.ndjson", 'w') as file:
        for line in poem:
            try:
                file.write("%s\n" %  line)
            except Exception as e:
#                 print('Write line error: ', e) 
                continue
    



        
if __name__ == '__main__':
    xml_to_string_tryout()