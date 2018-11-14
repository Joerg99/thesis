'''
Created on Nov 14, 2018

@author: joerg
'''
import codecs
import string
import os
def chicago_corpus_to_ndjson():
    one_file = []
    folder = '/home/joerg/workspace/thesis/gute_Daten/english_tagged/english_raw/'
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(folder+filename, 'r', encoding='latin-1') as file:
                for line in file:
                    line = line.strip()
                    line=bytes(line, 'utf-8').decode('ascii','ignore')
                    one_file.append(line)
            
            title_flag = False
            chicago_corpus=[]
            for line in one_file:
                if line.startswith('AUTHOR') or line.startswith('RHYME-POEM') or line in ['\n', '\r\n'] or line == '':
                    continue
            
                if line.startswith('TITLE'):
                    if len(line) < 7:
                        title = line
                    else:
                        title = line[6:]
                    stanza_no = 0
                    continue
                
                if line.startswith('RHYME'):
                    rhyme_schema = line[5:].replace(' ','')[::-1]
                    if '*' in rhyme_schema:
                        rhyme_schema  = 'a'*200
                    stanza_length= len(rhyme_schema)-1
                    stanza_no +=1
                    continue
                entry = [line, rhyme_schema[stanza_length] , stanza_no, title, filename[:-4]]
                print(entry)
                chicago_corpus.append(entry)
                stanza_length -=1
    print(len(chicago_corpus))
    with open('chicago_corpus.txt', 'w') as file:
        for line in chicago_corpus:
            file.write(str(line))

if __name__ == '__main__':
    chicago_corpus_to_ndjson()
