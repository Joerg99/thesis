'''
Created on 1 Nov 2018

@author: Jorg
'''
#===============================================================================
#  Data Structure 
#  english:
#  's', 'gid'
#
#  deutsch:
#  's' 'name / id', 'rhyme'
#
#===============================================================================

import ndjson
import xml.etree.ElementTree as etree
import os
import re
from bs4 import BeautifulSoup


def load_ndjson(file):
    with open(file) as f:
        data = ndjson.load(f)
    return data


#===============================================================================
# with open("data_file.json", "w") as write_file:
#     ndjson.dump(data, write_file)
#===============================================================================

#dta
def xml_to_string():
    # TODO Leerzeilen beseitigen
    #ALLE IN NICHT KAPUTT
    folder = '/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/gute Daten/german_tagged/Diachron_Sample_DTA_DTR_Rhyme_Annotated/'
    poem = []
    broken = 0
    for filename in os.listdir(folder):
        g_id = filename[:8]
        if filename.endswith('.xml'):
            try:
                root = etree.parse(folder+filename).getroot()
            except Exception as e:
                print(e)
                print(filename)
                break #continue
            for child in root[1].iter():      
                if child.tag == '{http://www.tei-c.org/ns/1.0}lg':
                    if 'rhyme' in child.attrib:
                        #print(child.attrib['rhyme'], len(child.attrib['rhyme']))
                        stanza_length = len(child.attrib['rhyme'])
                          
                        stanza = child
                          
                        i = 0
                        v = []
                        for verse in stanza.iter():
                            if verse.tag == '{http://www.tei-c.org/ns/1.0}l':
                                v.append(verse.text)
                                i +=1
                        if i == stanza_length:
                            #print(child.attrib['rhyme'])
                            for j in range(stanza_length):
                                try:
                                    poem.append('{"s": '+ '"'+v[j]+'", '+ '"rhyme": '+ '"'+child.attrib['rhyme'][j]+'", '+'"g_id": '+ '"'+str(g_id)+ '"'+'}')
                                except Exception as e:
                                    broken +=1
                                    print(e)
                                    print(v[j], child.attrib['rhyme'][j], str(g_id))
                                    continue
                                 
    print('kaputte Zeilen: ', broken)
    with open("dta_gold_alle_etree.ndjson", 'w') as file:
        for line in poem:
            try:
                file.write("%s\n" %  line)
            except: 
                continue
 
#dta_kaputt 
def beautifulsoup_for__kaputt():
    # for files in kaputt
    folder = '/home/joerg/workspace/thesis/gute_Daten/german_tagged/Diachron_Sample_DTA_DTR_Rhyme_Annotated/__kaputt/'
    #folder = '/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/gute Daten/german_tagged/Diachron_Sample_DTA_DTR_Rhyme_Annotated/__kaputt/'
    poems = []
    for file in os.listdir(folder):
        g_id = file[:8]
        if file.endswith('s40_TEI-P5.xml'):
            filename = folder+file
            with open(filename, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file)
                verses = []
                rhymes = ''
                for element in soup.find_all('lg', type='stanza'):
                    verses_temp = []
                    i = 0
 
                    for e in element.find_all('l'):
                        i +=1
                        verses_temp.append(str(e))
                     
                    if len(verses_temp) ==  len(element['rhyme']):
                        rhymes+=element['rhyme']
                        verses.extend(verses_temp)
                         
            if len(verses) == len(rhymes):
                verses = [re.sub(r'<[^>]*>', '', verse) for verse in verses]            
                verses = [re.sub(r'[":,\']', '', verse) for verse in verses]            
                for i in range(len(verses)):
                    poems.append('{"s": '+ '"'+verses[i]+'", '+ '"rhyme": '+ '"'+rhymes[i]+'", '+'"g_id": '+ '"'+str(g_id)+ '"'+'}')
 
    with open("dta_gold_kaputt_bs4_linuxtest.ndjson", 'w') as file:
            for line in poems:
                try:
                    file.write("%s\n" %  line)
                except: 
                    continue

#rap
def regex_rap():
    filename = '/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/gute Daten/german_tagged/hip_hop_complete_annotated/Absolute_Beginner-Das_Boot.xml'
    #filename = '/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/gute Daten/german_tagged/Diachron_Sample_DTA_DTR_Rhyme_Annotated/__kaputt/Czepko_Danielvon_1641_gold_p4_s12_TEI-P5.xml'
    root = etree.parse(filename).getroot()
    #for c in root[1][0][0][0][1]:
    #    print(c.text)
    rhymes  = re.findall(r'rhyme=(.+?)>', str(etree.tostring(root[1], encoding='utf-8')))
    rhymes = [re.sub(r' type="stanza"', '', rhyme)for rhyme in rhymes]
    rhymes = [re.sub(r'"', '', rhyme)for rhyme in rhymes]
    
    r_string = ''
    for rhyme in rhymes:
        r_string+=rhyme
    print(len(r_string))
    
    verses = re.findall(r'<ns0:l>(.+?)</ns0:l>', str(etree.tostring(root[1]), encoding='utf-8'))
    verses = [re.sub(r'<[^>]*>', '', verse) for verse in verses]
    print(len(verses))
    print(verses)
    with open("asdasdasd.txt", 'w') as file:
        for line in verses:
            file.write(line)


# rap, anti-k
def beautifulsoup_rap():
    #folder = '/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/gute Daten/german_tagged/hip_hop_complete_annotated/'
    folder = '/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/gute Daten/german_tagged/Korpus_Antikoerperchen_Reim_Annotiert/'
    poems = []
    for file in os.listdir(folder):
        g_id = file[:8]
        if file.endswith('.xml'):
            filename = folder+file
            with open(filename, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file)
                verses = []
                rhymes = ''
                for element in soup.find_all('lg', type='stanza'):
                    verses_temp = []
                    i = 0
 
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
 
    with open("antik.ndjson", 'w') as file:
            for line in poems:
                try:
                    file.write("%s\n" %  line)
                except: 
                    continue

#===============================================================================
# Manually DELETE TABS FROM NDJSON FILES!!!!!
#===============================================================================

if __name__ == '__main__':
    #xml_to_string()
    beautifulsoup_for__kaputt()
    #beautifulsoup_rap()
    #verse= "ksadj alk  sdj alksd,j a:sd "
    #print(re.sub(r'[:,a]', '', verse))
    #print(re.sub(r'  ', '__', verse))
    
    #data= load_ndjson('/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/JSON_Data/dta_gold_kaputt_bs4.ndjson')
    #data= load_ndjson('/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/JSON_Data/rap.ndjson')
    #data= load_ndjson('/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/JSON_Data/antik.ndjson')
    