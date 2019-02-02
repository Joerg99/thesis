'''
Created on Dec 8, 2018

@author: joerg
'''
from interface_2 import Corpus
from nltk.tokenize import RegexpTokenizer
import pprint
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import groupby

def make_four_line_stanza():
#     tg = Corpus('/home/joerg/workspace/thesis/Chicago/chicago.ndjson')
    tg = Corpus('/home/joerg/workspace/thesis/Deepspeare_Data/deepspeare_data.ndjson')
    corpusname = 'deepspeare'
    tg.filter_by_no_of_lines(4, 9999)
    
    # all_stanza hat alle stanza mit allen Laengen
    all_stanza = []
    for k, v in tg.data.items():
        temp = ([list(g) for k,g in groupby(v, lambda x: x[2])])
        all_stanza.extend(temp)
    
    
    def chunk_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    # all_quatrains hat nur noch stanza der Laenge vier
    all_quatrains = []
    
    for stanza in all_stanza:
        if len(stanza) >= 4:
            chunks = chunk_list(stanza, 4)
            for chunk in chunks:
                if len(chunk) == 4:
                    all_quatrains.append(chunk)
    
    print('anzahl stanza: ', len(all_quatrains))

    tokenizer = RegexpTokenizer(r'\w+')

    random.shuffle(all_quatrains)
    
    all_verses = []
    for stanza in all_quatrains:
        verse = ['sos']
        for line in stanza:
             
            token_in_verse = [string_replace_multiple(token).encode(encoding='ascii', errors='backslashreplace').decode('utf-8').lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
            verse.extend(token_in_verse)
            verse.append('xxxxxxxxxx')
        del verse[-1]
        for i in range(len(verse)):
            if verse[i] == 'xxxxxxxxxx':
                verse[i] = 'newline'# '<nl>' 
        all_verses.append(verse)
    flat_list = [token.lower() for stanza in all_verses for token in stanza]
    print(len(flat_list))
     
     
    percent_80 = int(len(flat_list) * 0.8)
    percent_90 = percent_80 + int(len(flat_list) * 0.1)
    
    # export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/'+corpusname+'/train_stanza.txt', 'w') as file:
        for i in range(percent_80):
            if flat_list[i+1] == 'sos':
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+str(flat_list[i+1])+'\n')
       
    # export test
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/'+corpusname+'/test_stanza.txt', 'w') as file:
        for i in range(percent_80, percent_90):
            if flat_list[i+1] == 'sos':
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+str(flat_list[i+1])+'\n')
    # export dev
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/'+corpusname+'/dev_stanza.txt', 'w') as file:
        for i in range(percent_90, len(flat_list)-1):
            if flat_list[i+1] == 'sos':
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+flat_list[i]+'\t'+str(flat_list[i+1])+'\n')
    print('done')
    
    
    
def create_single_verses():
#     tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups.ndjson')
#     tg = Corpus('/home/joerg/workspace/thesis/Deepspeare_Data/deepspeare_data.ndjson')
    tg = Corpus('/home/joerg/workspace/thesis/Chicago/chicago.ndjson')
    #tg.filter_sonnets()
    #print(len(tg.data))
#     tokenizer = RegexpTokenizer(r'\w+')
    all_verses = []
#     sos = '<sos>'
    #new_line = '<nl>'
    all_verses_as_list = []
    tokenizer = RegexpTokenizer(r'\w+')
    for value in tg.data.values():
#         poem = []
        for line in value:
            verse = ['sos']
            token_in_verse = [string_replace_multiple(token.lower()).encode(encoding='ascii', errors='backslashreplace').decode('utf-8') for token in tokenizer.tokenize(line[0]) if token.isalpha()]
            verse.extend(token_in_verse)
            all_verses_as_list.append(verse)
    
    random.shuffle(all_verses_as_list)
    all_verses = [word for verse in all_verses_as_list for word in verse ]
    
    print(len(all_verses)) 
    #export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/train.txt', 'w') as file:
        for i in range(600000):
            if all_verses[i+1] == 'sos':
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+str(all_verses[i+1])+'\n')
        
    # #export test
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/test.txt', 'w') as file:
        for i in range(600000, 690000):
            if all_verses[i+1] == 'sos':
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+str(all_verses[i+1])+'\n')
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/dev.txt', 'w') as file:
        for i in range(690000, len(all_verses)-1):
            if all_verses[i+1] == 'sos':
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+'<eos>'+'\n\n')
            else:
                file.write(str(i+1)+'\t'+all_verses[i]+'\t'+str(all_verses[i+1])+'\n')

############################

def string_replace_multiple(input_string):
    bad_char = [('ä', 'ae'), ('ö',  'oe'), ('ü', 'ue'),('ß', 'ss')]
    for i in range(len(bad_char)):
        input_string = input_string.replace(bad_char[i][0], bad_char[i][1])
    return input_string    

def change_embeddings():
    embeddings = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/embedding_textgrid_300.bin', delimiter = ' ', header=None)
    
    
    
    for i in range(len(embeddings[0])):
        embeddings.iat[i, 0] = string_replace_multiple(str(embeddings.iat[i, 0])).encode(encoding='ascii', errors='backslashreplace').decode('utf-8')

    
    embeddings.to_csv('embedding_textgrid_300_casing_ascii.bin', header= None, index= None, sep=' ', mode='w')
    print('done')


def make_embeddings_pos_neg():
    embeddings = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/embedding_textgrid_300_lower.bin', delimiter = ' ', header=None)
    embeddings_neg = embeddings.copy(deep=True)
    embeddings['posneg'] = 1.0
    
    
    for i in range(len(embeddings[0])):
        embeddings.iat[i, 0] = string_replace_multiple(str(embeddings.iat[i, 0])+'_p').encode(encoding='ascii', errors='backslashreplace').decode('utf-8')

    
    embeddings_neg['posneg'] = -1.0
    for i in range(len(embeddings_neg[0])):
        embeddings_neg.iat[i, 0] = string_replace_multiple(str(embeddings_neg.iat[i, 0])+'_n').encode(encoding='ascii', errors='backslashreplace').decode('utf-8')
    
    embeddings = pd.concat([embeddings, embeddings_neg])
    
    embeddings.to_csv('blah.txt', header= None, index= None, sep=' ', mode='w')
    print('done')


#exports file with extended _n or _p to duplicate words. Column with 1 and 0 not needed but doesn't hurt

def create_short_and_long_verses_embedding_duplicate():
    tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups.ndjson')
    tg.filter_by_no_of_lines(4, 9999)
    print(len(tg.data))
    all_verses_as_lists = [] # needed to shuffle
    all_verses = [] # needed for export to conll
    tokenizer = RegexpTokenizer(r'\w+')
    count_long_verses = 0
    for value in tg.data.values():
        for line in value:
            verse_len  = len(tokenizer.tokenize(line[0]))
            if verse_len >10:
                count_long_verses += 1
                verse = ['sos_n']
                token_in_verse = [string_replace_multiple(token+'_n').encode(encoding='ascii', errors='backslashreplace').decode('utf-8').lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
                verse.extend(token_in_verse)
                all_verses_as_lists.append(verse)
                #all_verses.extend(verse)
    
    for value in tg.data.values():
        for line in value:
            verse_len  = len(tokenizer.tokenize(line[0]))
            if verse_len < 5 and count_long_verses > 0:
                count_long_verses -= 1
                verse = ['sos_p']
                token_in_verse = [string_replace_multiple(token+'_p').encode(encoding='ascii', errors='backslashreplace').decode('utf-8').lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
                verse.extend(token_in_verse)
                all_verses_as_lists.append(verse)
                #all_verses.extend(verse)
    
    random.shuffle(all_verses_as_lists)
    all_verses = [word for verse in all_verses_as_lists for word in verse ]
    print(len(all_verses))

    #export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/pn_train_tg.txt', 'w') as file:
        for i in range(30000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos_p'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos_n'+'\t0'+'\n\n')
                    
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1])+'\t1'+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1])+'\t0'+'\n')
                    
    #export test
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/pn_test_tg.txt', 'w') as file:
        for i in range(30000, 40000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos_p'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos_n'+'\t0'+'\n\n')
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1])+'\t1 '+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1])+'\t0 '+'\n')
    
    # export dev
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/pn_dev_tg.txt', 'w') as file:
        for i in range(40000, 50000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos_p'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos_n'+'\t0'+'\n\n')
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1])+'\t1 '+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1])+'\t0 '+'\n')
    
    
    print('done')

def create_short_and_long_verses_embedding_duplicate_less_labels():
    tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups.ndjson')
    tg.filter_by_no_of_lines(4, 9999)
    print(len(tg.data))
    all_verses_as_lists = [] # needed to shuffle
    all_verses = [] # needed for export to conll
    tokenizer = RegexpTokenizer(r'\w+')
    count_long_verses = 0
    for value in tg.data.values():
        for line in value:
            verse_len  = len(tokenizer.tokenize(line[0]))
            if verse_len >8 and verse_len <13 :
                count_long_verses += 1
                verse = ['sos_n']
                token_in_verse = [string_replace_multiple(token+'_n').encode(encoding='ascii', errors='backslashreplace').decode('utf-8').lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
                verse.extend(token_in_verse)
                all_verses_as_lists.append(verse)
                #all_verses.extend(verse)
    
    for value in tg.data.values():
        for line in value:
            verse_len  = len(tokenizer.tokenize(line[0]))
            if verse_len < 5 and count_long_verses > 0:
                count_long_verses -= 1
                verse = ['sos_p']
                token_in_verse = [string_replace_multiple(token+'_p').encode(encoding='ascii', errors='backslashreplace').decode('utf-8').lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
                verse.extend(token_in_verse)
                all_verses_as_lists.append(verse)
                #all_verses.extend(verse)
    print('Anzahl aller Verse', len(all_verses_as_lists))
    random.shuffle(all_verses_as_lists)
    all_verses = [word for verse in all_verses_as_lists for word in verse ]
    print('Anzahl aller Wörter: ', len(all_verses))

    #export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/viel_pn_train_tg.txt', 'w') as file:
        for i in range(80000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t0'+'\n\n')
                    
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t1'+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t0'+'\n')
                    
    #export test
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/viel_pn_test_tg.txt', 'w') as file:
        for i in range(80000, 100000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t0'+'\n\n')
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t1 '+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t0 '+'\n')
    
    # export dev
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/viel_pn_dev_tg.txt', 'w') as file:
        for i in range(100000, 120000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t0'+'\n\n')
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t1 '+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t0 '+'\n')
    
#     train_data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/pn_train_tg.txt', delimiter='\t', header = None)
#     for i in range(len(train_data[2])):
#         train_data.iat[i, 2] = str(train_data.iat[i, 2][:-2])
#     
#     train_data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/pn_train_tg.txt', header= None, index= None, sep='\t', mode='w')
    
    print('done')


def create_short_and_long_verses_embedding_duplicate_concat_info():
    tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups.ndjson')
    tg.filter_by_no_of_lines(4, 9999)
    print(len(tg.data))
    all_verses_as_lists = [] # needed to shuffle
    all_verses = [] # needed for export to conll
    tokenizer = RegexpTokenizer(r'\w+')
    count_long_verses = 0
    for value in tg.data.values():
        for line in value:
            verse_len  = len(tokenizer.tokenize(line[0]))
            if verse_len >8 and verse_len <13 :
                count_long_verses += 1
                verse = ['sos_n']
                token_in_verse = [string_replace_multiple(token+'_n').encode(encoding='ascii', errors='backslashreplace').decode('utf-8').lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
                verse.extend(token_in_verse)
                all_verses_as_lists.append(verse)
                #all_verses.extend(verse)
    
    for value in tg.data.values():
        for line in value:
            verse_len  = len(tokenizer.tokenize(line[0]))
            if verse_len < 5 and count_long_verses > 0:
                count_long_verses -= 1
                verse = ['sos_p']
                token_in_verse = [string_replace_multiple(token+'_p').encode(encoding='ascii', errors='backslashreplace').decode('utf-8').lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
                verse.extend(token_in_verse)
                all_verses_as_lists.append(verse)
                #all_verses.extend(verse)
    print('Anzahl aller Verse', len(all_verses_as_lists))
    random.shuffle(all_verses_as_lists)
    all_verses = [word for verse in all_verses_as_lists for word in verse ]
    print('Anzahl aller Wörter: ', len(all_verses))

    #export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/viel_pn_train_tg.txt', 'w') as file:
        for i in range(80000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t0'+'\n\n')
                    
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t1'+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t0'+'\n')
                    
    #export test
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/viel_pn_test_tg.txt', 'w') as file:
        for i in range(80000, 100000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t0'+'\n\n')
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t1 '+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t0 '+'\n')
    
    # export dev
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/viel_pn_dev_tg.txt', 'w') as file:
        for i in range(100000, 120000):
            if all_verses[i+1] == 'sos_p' or all_verses[i+1] == 'sos_n':
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t1'+'\n\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+'eos'+'\t0'+'\n\n')
            else:
                if str(all_verses[i][-1]) == 'p':
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t1 '+'\n')
                else:
                    file.write(str(i+1)+'\t'+str(all_verses[i])+'\t'+str(all_verses[i+1][:-2])+'\t0 '+'\n')
    
#     train_data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/pn_train_tg.txt', delimiter='\t', header = None)
#     for i in range(len(train_data[2])):
#         train_data.iat[i, 2] = str(train_data.iat[i, 2][:-2])
#     
#     train_data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/pn_train_tg.txt', header= None, index= None, sep='\t', mode='w')
    
    print('done')


def plot_losses(path_name, title):
    with open(path_name, 'r') as file:        
        lines = file.read().splitlines()
    train = [float(value) for value in lines[0][:-1].split(' ')]
    dev = [float(value) for value in lines[1][:-1].split(' ')]
    test = [float(value) for value in lines[2][:-1].split(' ')]
    x_lab = []
    for i in range(1, len(train)+1):
        x_lab.append(i)
    plt.plot(x_lab, train, label='train')
    plt.plot(x_lab, dev, label='dev')
    plt.plot(x_lab, test, label='test')
    plt.legend()
    plt.title(title)
    plt.show()

#######
# this function prepares data to be translated into phonemes 
# after translation, the resulting file is read in by read_training_data_tag_alliteration
# and  converted to a dictionary.
#
# only called to export file for phoneme translation 
#######
def export_data_for_g2p_lookup():
#     tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups.ndjson')
    tg = Corpus('/home/joerg/workspace/thesis/Chicago/chicago.ndjson')
    #tg.filter_sonnets()
    #print(len(tg.data))
#     tokenizer = RegexpTokenizer(r'\w+')
    all_verses = []
#     sos = '<sos>'
    #new_line = '<nl>'
    tokenizer = RegexpTokenizer(r'\w+')
    for value in tg.data.values():
#         poem = []
        for line in value:
            verse = ['sos']
            token_in_verse = [string_replace_multiple(token).encode(encoding='ascii', errors='backslashreplace').decode('utf-8').lower() for token in tokenizer.tokenize(line[0]) if token.isalpha()]
            verse.extend(token_in_verse)
            all_verses.extend(verse)
    print(len(all_verses)) 
    #export train
    with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/chicago_aliteration.txt', 'w') as file:
        for i in range(len(all_verses)-1):
            if all_verses[i+1] == 'sos':
                file.write(all_verses[i]+'\n\n')
            else:
                file.write(all_verses[i]+'\n')

# reads in train/test/dev file 
# reads in g2p lookup table for one dataset
# returns train/dev/test file with additional alliteration tag

def read_training_data_and_tag_alliteration():
#     data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_wo_alits/train_stanza.txt', delimiter='\t' ,header=None, skip_blank_lines= False)
    data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/deepspeare/stanza_wo_alits/dev_stanza.txt', delimiter='\t' ,header=None, skip_blank_lines= False)

#     g2p_lookup_df = pd.read_csv('/home/joerg/workspace/thesis/Interface/aliteration_lookup_files/chicago_aliteration_converted', sep=' ', usecols = (0,1), header=None)
    g2p_lookup_df = pd.read_csv('/home/joerg/workspace/g2p_raw/g2p-seq2seq/deepspeare_aliteration_converted', sep=' ', usecols = (0,1), header=None)
    
    
    #make dictionary for lookup g2p 
    g2p_lookup = {}
    for i in range(len(g2p_lookup_df[0])):
        g2p_lookup[g2p_lookup_df.iat[i,0]] = g2p_lookup_df.iat[i,1]
    
        
    data[3] = ""  # column for first sounds
    data[4] = ""  # column for aliteration label
    for i in range(len(data[0])):
        try:
            data.set_value(i, 3, g2p_lookup[data.iat[i,1]])
        except:
            data.set_value(i, 3, " ")
                
    i = 0
    while i < len(data[0])-2:
        if data.iat[i, 3] == data.iat[i+1, 3] == data.iat[i+2, 3] and data.iat[i, 3].isalpha():
            data.iat[i, 4] = 1
            data.iat[i+1, 4] = 1
            data.iat[i+2, 4] = 1
            i +=3
        elif data.iat[i, 3] == data.iat[i+1, 3] and data.iat[i,3].isalpha():
            data.iat[i, 4] = 1
            data.iat[i+1, 4] = 1
            i +=2
        else:
            if isinstance(data.iat[i, 2], str):
                data.iat[i, 4] = 0
            i += 1
    
    print(data.loc[data[4] == 1])
    
    data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/deepspeare/stanza_w_alits/dev_stanza_alit.txt', sep='\t', columns=(0,1,2,4), header=None, index= False)




##### returns density of alliterations in a training quatrain and adds it in new column
def count_alliterations_in_training_data():
    data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/deepspeare/stanza_w_alits/dev_stanza_alit.txt', sep='\t', usecols = (0,1,2,3), header=None, skip_blank_lines=False)
    temp = []
    all_quatrains_in_tuples = []
    
    # all_quatrains... = list of list(quatrains) of tuples(alle infos) 
    for i in range(len(data[0])):
        if type(data.iat[i,1]) != float:
            temp.append((data.iat[i,0], data.iat[i,1], data.iat[i,2], data.iat[i,3]))
        else:
            all_quatrains_in_tuples.append(temp)
            temp = []
    
    counter_for_all_quatrains = []
    for quatrain in all_quatrains_in_tuples:
        counter = [quatrain[0][0], 0]          # count Anzahl von alliterierenden words und ersten index des quatrains
        for word in quatrain:
            if word[3] == 1:
                counter[1] += 1
        counter_for_all_quatrains.append(counter)

    data[4] = ""
    j = 0
    # adds value from counter to dataframe according to first index of the quatrain
    for i in range(len(data[0])):
        
        if data.iat[i, 0]==counter_for_all_quatrains[j][0]: # wenn index im dataframe == gespeicherter index im counter
            data.iat[i,4] = counter_for_all_quatrains[j][1]
                
            j+= 1
            if j == len(counter_for_all_quatrains):
                j-= 1
        if type(data.iat[i,1]) != float:
            data.iat[i, 4] = counter_for_all_quatrains[j-1][1]
    data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/deepspeare/stanza_w_alits_density/dev_stanza_alit_density.txt', sep='\t', columns=(0,1,2,3,4), header=None, index= False)

    counter_for_all_quatrains.sort(key=lambda tup: tup[1], reverse=True)
    print(counter_for_all_quatrains)


# returns number of alliterations in generates verses
def evaluate_generated_verses_on_aliterations():
    verse_with_alit = []
    verse_without_alit = []
    
    # make lookup table
#     g2p_lookup_df = pd.read_csv('/home/joerg/workspace/thesis/Interface/aliteration_lookup_files/chicago_aliteration_converted', sep=' ', usecols = (0,1), header=None)
    g2p_lookup_df = pd.read_csv('/home/joerg/workspace/thesis/Interface/aliteration_lookup_files/deepspeare_aliteration_converted', sep=' ', usecols = (0,1), header=None)
    g2p_lookup = {}
    for i in range(len(g2p_lookup_df[0])):
        g2p_lookup[g2p_lookup_df.iat[i,0]] = g2p_lookup_df.iat[i,1]
    
    # load generated verses
    with open('/home/joerg/Documents/Uni/30_Thesis/deepspeare_stanza_alit_100_level0_100_level12', 'r') as file:
        i = 0
        for line in file:
            if i <100:
                verse_with_alit.append(line.strip())
            else:
                verse_without_alit.append(line.strip())
            i += 1
            
    def count_alits(list_of_verses):
        verse_alits_phoneme = []
        for verse in list_of_verses:
            temp = []
            for token in verse.split(' '):
                try:
                    temp.append(g2p_lookup[token])
                except:
                    continue
            verse_alits_phoneme.append(temp)
        #print(verse_alits_phoneme)
        
        count_alits = 0
        for verse in verse_alits_phoneme:
            for i in range(len(verse)-1):
                if verse[i] == verse[i+1]:
                    count_alits += 1
        return count_alits
    
    print('with aliteration marker :', count_alits(verse_with_alit))
    print('without aliteration marker :', count_alits(verse_without_alit))
    
if __name__ == '__main__':
    print('asdads')
#     make_four_line_stanza()
#     read_training_data_and_tag_alliteration()
#     count_alliterations_in_training_data()

#     evaluate_generated_verses_on_aliterations()

#     create_single_verses()
#     export_data_for_g2p_lookup()
#     make_four_line_stanza()
#     create_short_and_long_verses_embedding_duplicate_less_labels()
#     plot_losses('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/models/test/plot_73_textgrid', '93000 verses, model: 256, 0.2 dropout ')
#     create_short_and_long_verses_concat_information()


