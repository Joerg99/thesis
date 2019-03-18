import sys
import pandas as pd
import pickle
import numpy as np

from os import listdir
from os.path import isfile, join

def make_bio_tagged_data():
    data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_w_alits_density_norm/train_stanza_alit_density_norm.txt', delimiter='\t' , usecols =(0,1,2), header=None, skip_blank_lines= False)
    g2p_lookup_df = pd.read_csv('/home/joerg/workspace/g2p_raw/g2p-seq2seq/chicago_aliteration_converted', sep=' ', usecols = (0,1), header=None)
    
    g2p_lookup = {}
    for i in range(len(g2p_lookup_df[0])):
        g2p_lookup[g2p_lookup_df.iat[i,0]] = g2p_lookup_df.iat[i,1]

    data[3] = ""  # column for first sounds
    data[4] = ""  # B I O 
    #### Add first sounds to column 3
    
    for i in range(len(data)):
        if type(data.iat[i,1]) == str:
            data.iat[i,4] =  'O'
        try:
            data.iat[i, 3] = g2p_lookup[data.iat[i,2]]
        except:
            if data.iat[i,2] == 'd':
                print('bloop', data.iat[i,2])
                data.iat[i, 3] =  'D'
            else:
                data.iat[i, 3] = " "
    ####
    
    #### add B I O Tags
    i = 0
    while i < len(data)-1:
        if data.iat[i,3] == data.iat[i+1,3]:
            data.iat[i,4] = 'B'
            data.iat[i+1,4] = 'I'
            i+=2
        else: 
            i+=1
    for i in range(len(data)-1):
        if data.iat[i,3] == data.iat[i+1, 3] and data.iat[i,4] == 'I':
            data.iat[i+1, 4] = 'I'
    ####
    data[5] = data[4]
    

    data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/train.txt', sep='\t', columns=(0,1,2,4,5), header=None, index= False)



def export_conll_file_for_rhyme_evaluation(number_of_quatrains_per_file):
    data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/train.txt', delimiter='\t' , usecols =(0,1,2,3), header=None, skip_blank_lines= False)
    quatrains_export= []
    quatrain = ''
    for i in range(len(data)):
        if type(data.iat[i,1]) == str:
            quatrain += data.iat[i,1]+' '
        else:
            quatrains_export.append(quatrain)
            quatrain = ''
    quatrains_export.append(quatrain)

    quatrains_chunked = [quatrains_export[x:x+number_of_quatrains_per_file] for x in range(0, len(quatrains_export), number_of_quatrains_per_file)]
    
    for i in range(len(quatrains_chunked)):
        with open('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/evaluation_files/chicago/mtl/'+str(i)+'_quatrains_train_all', 'w') as file:
            for line in quatrains_chunked[i]:
                file.write('%s<eos>\n' %line)
            
    
# read in training data
# read in file of rhyming words pairs
# add column to training data with rhyme tags
 
def add_rhyme_tags_to_bio_data():
    
    # load training data file
    data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/test.txt', delimiter='\t' , usecols =(0,1,2,3,4), header=None, skip_blank_lines= False)
    
    # load rhyming word pairs
    mypath = '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/evaluation_files/chicago/mtl/test'
    files_to_evaluate = [join(mypath,file) for file in listdir(mypath) if isfile(join(mypath, file))]
    files_to_evaluate = sorted(files_to_evaluate)
    
    print(files_to_evaluate)
    
    all_rhyme_pairs_joined = []    
    for f in files_to_evaluate:
        with open(f, 'rb') as file:
            rhyme_pairs = pickle.load(file)
        all_rhyme_pairs_joined.extend(rhyme_pairs)
    print(len(all_rhyme_pairs_joined))
    
    begin_of_quatrains = []
    for i in range(len(data)):
        if data.iat[i,1] == 'sos':
            begin_of_quatrains.append(i)
    begin_of_quatrains.append(len(data))
    
#     for quatrain in all_rhyme_pairs_joined[:30]:
#         print(quatrain)
    data[5] = ''

    # 6 ist erst nur Helfer
    data[6] = np.nan
    verse_marker = 1
    for i in range(len(data)):
        if data.iat[i,2] == 'newline':
            data.iat[i,6] = verse_marker
            verse_marker+=1
        elif data.iat[i,0] != data.iat[i,0]:
            verse_marker = 1
        else:
            data.iat[i,6] = verse_marker
    
    for i in range(len(all_rhyme_pairs_joined)):               # for each quatrain
        start_range = begin_of_quatrains[i]      
        end_range = begin_of_quatrains[i+1]      
         
        for j in range(len(all_rhyme_pairs_joined[i])):                # for each rhyming pair in a quatrain
            pos_s1 = -1
            pos_s2 = -1
            for k in range(start_range, end_range):         # iterate over range of a  quatrains
                if data.iat[k,2] == all_rhyme_pairs_joined[i][j][0]:   # if first  word is found
                    pos_s1 = k                                          # merke position
                    #data.iat[k,5] = 'S1'
                if data.iat[k,2] == all_rhyme_pairs_joined[i][j][1]:   #if second word is found and pos s1 is already found
                    if pos_s1 != -1:
                        pos_s2 = k
                        break
                    #data.iat[k,5] = 'S2'
            if pos_s1 != -1 and pos_s2 != -1:
                if data.iat[pos_s1, 6] == data.iat[pos_s2, 6]:
                    dist = np.abs(pos_s1 - pos_s2)
                    data.iat[pos_s1, 5] = 'S+'+str(dist)
                    data.iat[pos_s2, 5] = 'S-'+str(dist)
    
    for i in range(len(data)):
        if data.iat[i,5] == '':
            if data.iat[i,0] == data.iat[i,0]:
                data.iat[i,5] = 0
    
    data[6] = data[5]
    data[5] = data[5].shift(1)
    data[3] = data[3].shift(1)
    
    for i in range(len(data)):
        if data.iat[i,1] == 'sos':
            data.iat[i,3] = 'start'
            data.iat[i,5] = 'start'
    print(data)
    
    data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/test_rhyme_tagged_shift.txt', sep='\t', columns=(0,1,2,3,4,5,6), header=None, index= False)



# def add_start_tags_to_already_tagged_data():
#     data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/dev_rhyme_tagged.txt', delimiter='\t' , usecols =(0,1,2,3,4,5,6), header=None, skip_blank_lines= False)
#     
#     for i in range(len(data)):
#         if data.iat[i,1] =='sos':
#             data.iat[i,3] = 'start'
#             data.iat[i,4] = 'start'
#             data.iat[i,5] == 'start'
#             data.iat[i,6] = 'start'
#         if data.iat[i,0] != data.iat[i,0]:
#             data.iat[i,3] = np.nan
#             data.iat[i,5] = np.nan
#     print(data)
#     data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/test_rhyme_tagged_shift.txt', sep='\t', columns=(0,1,2,3,4,5,6), header=None, index= False)




# def shift_mtl_side_info_to_input_and_label():
#     data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/dev_rhyme_tagged.txt', delimiter='\t' , usecols =(0,1,2,3,4,5,6), header=None, skip_blank_lines= False)
#     data[3] = data[3].shift(1)
#     for i in range(len(data)):
#         if data.iat[i,5] == 'start':
#             data.iat[i,4] = 'O'
#             data.iat[i,3] = 'start'
#     for i in range(len(data)-1):
#         if data.iat[i,3] =='start':
#             data.iat[i+1,3] = data.iat[i,4]
# 
#     data[5] = data[5].shift(1)
#     for i in range(len(data)):
#         if data.iat[i,3] == 'start':
#             data.iat[i,6] = '0'
#             data.iat[i,5] = 'start'
#     for i in range(len(data)-1):
#         if data.iat[i,5] =='start':
#             data.iat[i+1,5] = data.iat[i,6]
#     for i in range(len(data)):
#         if data.iat[i,0] != data.iat[i,0]:
#             data.iat[i,3] = np.nan
#             data.iat[i,4] = np.nan
#             data.iat[i,6] = np.nan
#             data.iat[i,5] = np.nan
#     data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/dev_rhyme_tagged_shift.txt', sep='\t', columns=(0,1,2,3,4,5,6), header=None, index= False)

def add_end_rhyme_tags():
    data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/test_rhyme_tagged_shift.txt', delimiter='\t' , usecols =(0,1,2,3,4,5,6), header=None, skip_blank_lines= False)
    print(data[30:80])
    
    verse_end_words = []
    all_verse_end_words = []
    
    for i in range(len(data)):
        if data.iat[i,2] in ['newline', '<eos>']:
            verse_end_words.append((data.iat[i,1], i))
        if data.iat[i,2] != data.iat[i,2]:
            all_verse_end_words.append(verse_end_words)
            verse_end_words = []
            
            
    if len(verse_end_words) != 0:
        all_verse_end_words.append(verse_end_words)
    print(all_verse_end_words[-10:])

    for q in all_verse_end_words:
        index_1 = 0
        index_2 = 0
        index_3 = 0
        index_4 = 0
        last_two = False
        if q[0][0][-2:] == q[1][0][-2:]:
            last_two = True
            index_1 = 1
            index_2 = -1
            index_3 = 1
            index_4 = -1
        elif q[0][0][-2:] == q[2][0][-2:]:
            last_two = True
            index_1 = 2
            index_2 = 2
            index_3 = -2
            index_4 = -2
        elif q[0][0][-2:] == q[3][0][-2:]:
            last_two = True
            index_1 = 3
            index_2 = 1
            index_3 = -1
            index_4 = -3
        
        if last_two == False:        
            if q[0][0][-1] == q[1][0][-1]:
                index_1 = 1
                index_2 = -1
                index_3 = 1
                index_4 = -1
            elif q[0][0][-1] == q[2][0][-1]:
                index_1 = 2
                index_2 = 2
                index_3 = -2
                index_4 = -2
            elif q[0][0][-1] == q[3][0][-1]:
                index_1 = 3
                index_2 = 1
                index_3 = -1
                index_4 = -3
        
        if index_1 == 0:
            if q[1][0][-1:] == q[2][0][-1:]:
                last_two = True
                index_1 = 3
                index_2 = 1
                index_3 = -1
                index_4 = -3
            elif q[1][0][-1:] == q[3][0][-1:]:
                last_two = True
                index_1 = 2
                index_2 = 2
                index_3 = -2
                index_4 = -2

        if index_1 == 0:
            if q[2][0][-1:] == q[3][0][-1:]:
                last_two = True
                index_1 = 1
                index_2 = -1
                index_3 = 1
                index_4 = -1
        
        
        data.iat[q[0][1],  5  ] = 'O'+str(index_1) if str(index_1).startswith('-') else 'O+'+str(index_1)
        data.iat[q[1][1],  5  ] = 'O'+str(index_2) if str(index_2).startswith('-') else 'O+'+str(index_2)
        data.iat[q[2][1],  5  ] = 'O'+str(index_3) if str(index_3).startswith('-') else 'O+'+str(index_3)
        data.iat[q[3][1],  5  ] = 'O'+str(index_4) if str(index_4).startswith('-') else 'O+'+str(index_4)
    
    data[6] = data[5].shift(-1)
    for i in range(len(data)):
        if data.iat[i,6] != data.iat[i,6]:
            data.iat[i,6] = 0
        if data.iat[i,6] == 'start':
            data.iat[i,6] = np.nan
        if data.iat[i,0] != data.iat[i,0]:
            data.iat[i,3] = np.nan
            data.iat[i,5] = np.nan
        
    print(data[30:80])
    data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_bio/test_rhyme_tagged_shift_2.txt', sep='\t', columns=(0,1,2,3,4,5,6), header=None, index= False)

if __name__ == '__main__':
    print('asda')
#     make_bio_tagged_data()
    
#     export_conll_file_for_rhyme_evaluation(20000)
    add_rhyme_tags_to_bio_data()
    add_end_rhyme_tags()
    
#     add_start_tags_to_already_tagged_data()
#     shift_mtl_side_info_to_input_and_label()

    print('done')
    