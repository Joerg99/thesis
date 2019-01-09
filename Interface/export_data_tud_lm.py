'''
Created on Dec 8, 2018

@author: joerg
'''
from interface_2 import Corpus
from nltk.tokenize import RegexpTokenizer
import pprint
import pandas as pd


def make_four_line_stanza():
    tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups.ndjson')
    tg.filter_by_no_lines(4, 9999)
    print(len(tg.data))
    tokenizer = RegexpTokenizer(r'\w+')
    
    temp = []
    quadruplets = []
    for poem in tg.data.values():
        for line in poem:
            if len(temp) == 0 or line[2] == temp[0][2]:
                temp.append(line)
            else:
                temp = []
                temp.append(line)
            
            if len(temp) == 4:
                quadruplets.append(temp)
                temp = []
    print(len(quadruplets))
    quadruplets = quadruplets[:200000] # reduce amount of quadruplets
    
    all_verses = []
    for stanza in quadruplets:
        verse = ['startseq']
        for line in stanza:
            token_in_verse = [token.lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
            verse.extend(token_in_verse)
            verse.append('xxxxxxxxxx')
        del verse[-1]
        for i in range(len(verse)):
            if verse[i] == 'xxxxxxxxxx':
                verse[i] = 'newline'# '<nl>' 
        all_verses.append(verse)
    flat_list = [token.lower() for stanza in all_verses for token in stanza]
    print(len(flat_list))
    
    
    # export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/train_stanza.txt', 'w') as file:
        for i in range(200000):
            if flat_list[i+1] == 'startseq':
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+str(flat_list[i+1])+'\n')
     
    # export test
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/test_stanza.txt', 'w') as file:
        for i in range(300000, 325000):
            if flat_list[i+1] == 'startseq':
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+str(flat_list[i+1])+'\n')
    # export dev
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/dev_stanza.txt', 'w') as file:
        for i in range(325000, 350000):
            if flat_list[i+1] == 'startseq':
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+str(flat_list[i+1])+'\n')

def create_single_verses():
    tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups.ndjson')
    tg.filter_by_no_lines(4, 9999)
    print(len(tg.data))
#     tokenizer = RegexpTokenizer(r'\w+')
    all_verses = []
#     sos = '<sos>'
    #new_line = '<nl>'
    tokenizer = RegexpTokenizer(r'\w+')
    for value in tg.data.values():
#         poem = []
        for line in value:
            verse = ['sos']
            token_in_verse = [token.lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
            verse.extend(token_in_verse)
            all_verses.extend(verse)
     
    #export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/train_tg.txt', 'w') as file:
        for i in range(60000):
            if all_verses[i+1] == 'startSeq':
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+str(all_verses[i+1])+'\n')
      
    # #export test
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/test_tg.txt', 'w') as file:
        for i in range(60000, 80000):
            if all_verses[i+1] == 'startSeq':
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+str(all_verses[i+1])+'\n')

############################

def string_replace_multiple(input_string):
    bad_char = [('ä', 'ae'), ('ö',  'oe'), ('ü', 'ue'),('ß', 'ss')]
    for i in range(len(bad_char)):
        input_string = input_string.replace(bad_char[i][0], bad_char[i][1])
    return input_string    

def make_embeddings_pos_neg():
    embeddings = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/embedding_textgrid_300_lower.bin', delimiter = ' ', header=None)
    embeddings_neg = embeddings.copy(deep=True)
    embeddings['posneg'] = 1.0
    
    
    for i in range(len(embeddings[0])):
        embeddings.iat[i, 0] = string_replace_multiple(str(embeddings.iat[i, 0])+'_p')

    
    embeddings_neg['posneg'] = -1.0
    for i in range(len(embeddings_neg[0])):
        embeddings_neg.iat[i, 0] = string_replace_multiple(str(embeddings_neg.iat[i, 0])+'_n')
    
    embeddings = pd.concat([embeddings, embeddings_neg])
    
    embeddings.to_csv('blah.txt', header= None, index= None, sep=' ', mode='w')
    print('done')



# used fpr test if conditioned generation can work
def create_short_and_long_verses():
    tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups.ndjson')
    tg.filter_by_no_of_lines(4, 9999)
    print(len(tg.data))
#     tokenizer = RegexpTokenizer(r'\w+')
    all_verses = []
#     sos = '<sos>'
#     new_line = '<nl>'
    tokenizer = RegexpTokenizer(r'\w+')
    pos_verses = 0
    neg_verses = 0
    for value in tg.data.values():
#         poem = []
        for line in value:
            verse_len  = len(tokenizer.tokenize(line[0]))
            if verse_len < 5:
                pos_verses += 1
                verse = ['sos_p']
                token_in_verse = [string_replace_multiple(token+'_p') for token in tokenizer.tokenize(line[0]) if token.isalpha()]
                verse.extend(token_in_verse)
                
                all_verses.extend(verse)
            elif verse_len >10:
                neg_verses += 1
                verse = ['sos_n']
                token_in_verse = [string_replace_multiple(token+'_n') for token in tokenizer.tokenize(line[0]) if token.isalpha()]
                verse.extend(token_in_verse)
                all_verses.extend(verse)
    #export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/train_tg.txt', 'w') as file:
        for i in range(30000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+'<eos>'+'\t1'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+str(all_verses[i+1])+'\t1'+'\n')
       
    #export test
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/test_tg.txt', 'w') as file:
        for i in range(30000, 40000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+'<eos>'+'\t1'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+str(all_verses[i+1])+'\t1 '+'\n')
    print('done')

    
if __name__ == '__main__':
    create_short_and_long_verses()
    #make_embeddings_pos_neg()