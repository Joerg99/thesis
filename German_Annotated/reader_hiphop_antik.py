'''
Created on Nov 10, 2018

@author: joerg
'''
import ndjson
import xml.etree.ElementTree as etree
import os
import re
from bs4 import BeautifulSoup



# read in Hiphop data und ANTIK ---> GEHT!
def beautifulsoup_rap():
#     folder = '/home/joerg/workspace/thesis/gute_Daten/german_tagged/Korpus_Antikoerperchen_Reim_Annotiert/'
    folder = '/home/joerg/workspace/thesis/gute_Daten/german_tagged/hip_hop_complete_annotated/'
    poems = []
    for file in os.listdir(folder):
        g_id = file[:8]
        if file.endswith('.xml'):
            filename = folder+file
            with open(filename, 'r', encoding='utf-8') as file:
                try:
                    soup = BeautifulSoup(file)
                except Exception as e:
                    print(filename)
                    continue
                verses = []
                rhymes = ''
                for element in soup.find_all('lg', type='stanza'):
                    verses_temp = []
                    i = 0
                    
                    # checkt ob Reimschema gleiche Länge hat wie Anzahl der Zeilen
                    # ABER manche Strophen Reimen sich nicht und werden daher nicht übernommen
                    for e in element.find_all('l'):
                        i +=1
                        verses_temp.append(str(e))
                     
                    if len(verses_temp) ==  len(element['rhyme']):
                        rhymes+=element['rhyme']
                        verses.extend(verses_temp)
                         
            if len(verses) == len(rhymes):
                verses = [re.sub(r'<[^>]*>', '', verse) for verse in verses]            
                verses = [re.sub(r'["]', ' ', verse) for verse in verses]            
                verses = [re.sub(r'\\t', ' ', verse) for verse in verses]            
                for i in range(len(verses)):
                    poems.append('{"s": '+ '"'+verses[i]+'", '+ '"rhyme": '+ '"'+rhymes[i]+'", '+'"g_id": '+ '"'+str(g_id)+ '"'+'}')
 
    with open("hiphop.ndjson", 'w') as file:
            for line in poems:
                try:
                    file.write("%s\n" %  line)
                except: 
                    continue

    


if __name__ == '__main__':
    beautifulsoup_rap()