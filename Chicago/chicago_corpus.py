'''
Created on Nov 14, 2018

@author: joerg
'''
import os
import re


def chicago_corpus_to_ndjson():
    folder = '/home/joerg/workspace/thesis/Chicago/english_tagged/english_raw/'
    
    auth14 = []
    auth15 = []
    auth16 = []
    auth17 = []
    auth18 = []
    for filename in os.listdir(folder):
        if filename.endswith('.authors'):
            with open(folder+filename, 'r')as file:
                if filename[:2] == '14':
                    for line in file:
                        auth14.append(line.strip())
                if filename[:2] == '15':
                    for line in file:
                        auth15.append(line.strip())
                if filename[:2] == '16':
                    for line in file:
                        auth16.append(line.strip())
                if filename[:2] == '17':
                    for line in file:
                        auth17.append(line.strip())
                if filename[:2] == '18':
                    for line in file:
                        auth18.append(line.strip())
                
    one_file = []
    poem_no__ = 0
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(folder+filename, 'r', encoding='latin-1') as file:
                for line in file:
                    line = line.strip()
                    line=bytes(line, 'utf-8').decode('ascii','ignore')
                    one_file.append(line)
                date = ''
                if filename[:-4] in auth14:
                    date +='1400'
                elif filename[:-4] in auth15:
                    date+='1500'
                elif filename[:-4] in auth16:
                    date+='1600'
                elif filename[:-4] in auth17:
                    date+='1700'
                else:
                    date+='1800'
                print(date)
                chicago_corpus=[]
                for line in one_file:
                    if line.startswith('AUTHOR') or line.startswith('RHYME-POEM') or line in ['\n', '\r\n'] or line == '':
                        continue
                
                    if line.startswith('TITLE'):
                        poem_no__ += 1
                        #print(filename, line, poem_no)
                        stanza_no = 0
                        continue
    #                     if len(line) < 7:
    #                         title = 'unknown'
    #                     else:
    #                         title = line[6:]
                    
                    if line.startswith('RHYME'):
                        rhyme_schema = line[5:].replace(' ','')[::-1]
                        if '*' in rhyme_schema:
                            rhyme_schema  = 'aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz'*10
                        stanza_length= len(rhyme_schema)-1
                        stanza_no +=1
                        continue
                    
                    #entry = [line, rhyme_schema[stanza_length] , stanza_no, title, filename[:-4]]
                    export_line = '{"s": '+ '"'+line.strip()+'", '+ '"rhyme": ' + '"' + rhyme_schema[stanza_length]+'", ' + '"poem_no": ' + '"' + str(poem_no__-24705)+'", ' + '"stanza_no": '+ '"'+str(stanza_no)+ '", '+ '"released": ' + '"' + str(date)+'"' +'}'
    
                    chicago_corpus.append(export_line)
                    stanza_length -=1   
                
    print(len(chicago_corpus))
    with open('chicago_corpus.ndjson', 'w') as file:
        for line in chicago_corpus:
            file.write("%s\n" %  str(line))


def chicago_2():
    
    folder = '/home/joerg/workspace/thesis/Chicago/english_tagged/english_raw/'
    auth14 = []
    auth15 = []
    auth16 = []
    auth17 = []
    auth18 = []
    for filename in os.listdir(folder):
        if filename.endswith('.authors'):
            with open(folder+filename, 'r')as file:
                if filename[:2] == '14':
                    for line in file:
                        auth14.append(line.strip())
                if filename[:2] == '15':
                    for line in file:
                        auth15.append(line.strip())
                if filename[:2] == '16':
                    for line in file:
                        auth16.append(line.strip())
                if filename[:2] == '17':
                    for line in file:
                        auth17.append(line.strip())
                if filename[:2] == '18':
                    for line in file:
                        auth18.append(line.strip())
    
    
    poem_no = 0
    all_lines = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            one_file = []
            with open(folder+filename, 'rb') as file:
                for line in file:
                    line = line.strip().decode('utf-8', 'ignore').encode('utf-8')
                    line = ''.join([char for char in line.decode('utf-8') if not char.isdigit()])
                    line = line.strip()
                    line = ' '.join(line.split())
                    if len(line) > 1 and not line.startswith('AUTHOR') and not line.startswith('RHYME-POEM'):
                        one_file.append(line)
                date = ''
                if filename[:-4] in auth14:
                    date +='1400'
                elif filename[:-4] in auth15:
                    date+='1500'
                elif filename[:-4] in auth16:
                    date+='1600'
                elif filename[:-4] in auth17:
                    date+='1700'
                else:
                    date+='1800'
            #split one file to one list per poem. poem = [Title, Rhyme, a,b,c, Rhyme, a,b....]
            # poems = all poems of one file
            poems = []
            poem = []
            for line in one_file:
                if line.startswith('TITLE'):
                    if len(poem) > 0 :
                        poems.append(poem)
                    poem = []
                poem.append(line)
            poems.append(poem)
            
            #print(poem)
            # input poem
            # output list of lists each element in it is  rhyme + stanza. one lol is one poem 
            def cluster_stanza_in_poem(poem):
                clustered_file = []
                rhyme_position = []
                for i in range(len(poem)):
                    if poem[i].startswith('RHYME'):
                        rhyme_position.append(i)
                for i in range(len(poem)):
                    try:
                        stanza = poem[rhyme_position[i]:rhyme_position[i+1]]
                        clustered_file.append(stanza)
                    except:
                        stanza = poem[rhyme_position[i]:len(poem)]
                        clustered_file.append(stanza)
                        #print(filename)
                        break
                #print(rhyme_position)
                for element in clustered_file:
                    for i in range(len(element)):
                        if element[i].startswith('RHYME'):
                            element[i] = element[i][5:].replace(' ','')
                return clustered_file
            
            stanzas_in_poems = []
            for poem in poems:
                stanzas_in_poems.append(cluster_stanza_in_poem(poem))
            for i  in range(len(stanzas_in_poems)): # poem
                poem_no += 1
                stanza_no = 0
                for j in range(len(stanzas_in_poems[i])): # stanza
                    stanza_no += 1
                    #print(stanzas_in_poems[i][j]) #stanzas_in_poems[i][j] == stanza
                    
                    # Reimschama == Anzahl Verse
                    if len(stanzas_in_poems[i][j][0]) == len(stanzas_in_poems[i][j])-1:
                        schema = stanzas_in_poems[i][j].pop(0)
                        for k in range(len(stanzas_in_poems[i][j])):
                            verse = re.sub(r'["\'\-\\\/:,\']','', stanzas_in_poems[i][j][k].strip())
                            export_line = '{"s": '+ '"'+ verse +'", '+ '"rhyme": ' + '"' + schema[k] +'", ' + '"poem_no": ' + '"' + str(poem_no)+'", ' + '"stanza_no": '+ '"'+str(stanza_no)+ '", '+ '"released": ' + '"' + str(date)+'"' +'}'
                            all_lines.append(export_line)
                            
                    # Reimschema == aa*
                    elif stanzas_in_poems[i][j][0] == 'aa*':
                        schema = 'aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz'*5
                        del(stanzas_in_poems[i][j][0])
                        for k in range(len(stanzas_in_poems[i][j])):
                            verse = re.sub(r'["\'\-\\\/:,\']','', stanzas_in_poems[i][j][k].strip())
                            export_line = '{"s": '+ '"'+ verse +'", '+ '"rhyme": ' + '"' + schema[k] +'", ' + '"poem_no": ' + '"' + str(poem_no)+'", ' + '"stanza_no": '+ '"'+str(stanza_no)+ '", '+ '"released": ' + '"' + str(date)+'"' +'}'
                            all_lines.append(export_line)

                    # Reimschema 'ab*'
                    elif '*' in stanzas_in_poems[i][j][0]:
                        schema = stanzas_in_poems[i][j][0][:-1]*100
                        del(stanzas_in_poems[i][j][0])
                        for k in range(len(stanzas_in_poems[i][j])):
                            verse = re.sub(r'["\'\-\\\/:,\']','', stanzas_in_poems[i][j][k].strip())
                            export_line = '{"s": '+ '"'+ verse +'", '+ '"rhyme": ' + '"' + schema[k] +'", ' + '"poem_no": ' + '"' + str(poem_no)+'", ' + '"stanza_no": '+ '"'+str(stanza_no)+ '", '+ '"released": ' + '"' + str(date)+'"' +'}'
                            all_lines.append(export_line)
    
    with open('chicago.ndjson', 'w') as file:
        for line in all_lines:
            file.write('%s\n' % line)

def reader():
    folder= '/home/joerg/workspace/thesis/Chicago/english_tagged/english_raw/'
    data = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(folder+filename, 'rb') as file:
                for line in file:
                    if len(line) > 1:
                        print(len(line))
                        data.append(line.strip().decode('latin-1').encode('utf-8'))
    return data        
    
if __name__ == '__main__':
#     chicago_corpus_to_ndjson()
    chicago_2()
#     data = reader()
#     for element in data:
#         print(element.decode('utf-8'))