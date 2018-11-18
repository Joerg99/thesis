'''
Created on Nov 15, 2018

@author: joerg
'''

def deepspeare_to_ndjson():
    poem_no = 0
    all_lines = []
    with open('/home/joerg/workspace/thesis/Deepspeare_Data/english_untagged/Deepspeare_sonnet/sonnet_complete.txt', 'r') as file:
        for line in file:
            poem_no += 1
            stanza_no = 1
            verse =''
            verse_count = 0
            for word in line.split():
                if word == '<eos>':
                    verse_count+=1
                    if verse_count in [5,9,13]:
                        stanza_no += 1
                    #print(verse, poem_no, stanza_no)
                    verse = verse.replace('"', '')
#                     export_line = '{"s": '+ '"'+verse.strip()+'", ' + '"stanza_no": '+'"'+str(stanza_no)+'", '+'"poem_no": '+ '"'+str(poem_no)+ '"'+'}'
                    export_line = '{"s": '+ '"'+verse.strip()+'", '+ '"rhyme": ' + '"' + '__'+'", ' + '"poem_no": ' + '"' + str(poem_no)+'", ' + '"stanza_no": '+ '"'+str(stanza_no)+ '", '+ '"released": ' + '"' + '0000'+'"' +'}'

                    
                    all_lines.append(export_line)
                    verse=''
                else:
                    verse+=word+' '
    
    with open('deepspeare_data.ndjson', 'w') as file:
        for line in all_lines:
            file.write("%s\n" %  line)
    
if __name__ == '__main__':
    deepspeare_to_ndjson()