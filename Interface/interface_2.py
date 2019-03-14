'''
Created on Dec 6, 2018

@author: joerg
'''
import ndjson
from pprint import pprint
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pyphen
import epitran
import nltk
from collections import Counter
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from itertools import islice
from aylienapiclient import textapi
import sys
import random
import pandas as pd
import time
sys.path.insert(0, '/home/joerg/workspace/g2p-seq2seq/g2p_seq2seq')

import matplotlib.pyplot as plt

import g2p_seq2seq


class Corpus:
    def __init__(self, filename):
        self.data = self.__load_data(filename)
        
    def __load_data(self, filename):
        with open(filename, 'rb') as file:
            data = ndjson.load(file)
        print('Data loaded')
        
        data_dict = {}    
        poem_ids = set()
        for line in data:
            poem_ids.add(line['poem_no'])
        for p_id in poem_ids:
            data_dict[p_id] = []
            
        for line in data:
            data_dict[line['poem_no']].append((line['s'], line['rhyme'], line['stanza_no'], line['released'], line['author']))
        return data_dict
    
    # filter poems by certain number of lines
    def filter_by_no_of_lines(self, min_no_lines, max_no_lines):
        temp = {}
        for k, v in self.data.items():
            if len(v) >= min_no_lines and len(v)< max_no_lines:
                temp[k] = v
        self.data = temp
        
    # filter by verse length
    def filter_by_verse_length(self, min_average_length, max_average_length):
        tokenizer = RegexpTokenizer(r'\w+')
        temp = {}
        for k, v in self.data.items():
            no_token = 0
            for verse in v:
                no_token += len(tokenizer.tokenize(verse[0]))
            poem_average_length = no_token / len(v)
            if poem_average_length < max_average_length and  poem_average_length > min_average_length:
                temp[k] = v
        self.data = temp
    
    #uses number of lines filter and average verse length filter to get sonnets only (14 line, average verse length 9-13)
    def filter_sonnets(self):
        self.filter_by_no_of_lines(14,15)
        self.filter_by_verse_length(9,13)

    def filter_by_year(self, from_year, to_year):
        temp = {}
        for k,v in self.data.items():
            if int(v[0][3]) > from_year and int(v[0][3]) <= to_year:
                temp[k] = v
        self.data = temp

    
    def filter_limericks(self):
        self.filter_by_no_of_lines(5,6)
        tokenizer = RegexpTokenizer(r'\w+')
        temp = {}
        for k, v in self.data.items():
            no_token_long = 0
            no_token_short = 0
            for i in range(len(v)):
                if i in [0,1,4]:
                    no_token_long += len(tokenizer.tokenize(v[i][0]))
                else:
                    no_token_short += len(tokenizer.tokenize(v[i][0]))
            if (no_token_long / 3 ) < (no_token_short / 2):
                temp[k] = v
        self.data = temp
        

    def filter_by_author(self, author):
        temp = {}
        for k,v in self.data.items():
            if v[0][4] == author:
                temp[k] = v
        self.data = temp

    def syllablificate(self, language='en_EN'):
        if language == 'de':
            language = pyphen.Pyphen(lang='de_DE') # lang='de_DE' oder 'en_EN'
        else:
            language = pyphen.Pyphen(lang='en_EN') # lang='de_DE' oder 'en_EN'
        temp_dictionary = {}
        for k, v in self.data.items():
            temp_value = []
            for line in v:
                temp_verse = ''
                for word in line[0].split():
                    temp_verse += language.inserted(word+' ')
                line = list(line)
                line[0] = temp_verse
                line = tuple(line)
                temp_value.append(line)    
            temp_dictionary[k] = temp_value
        self.data = temp_dictionary
    
    def to_phoneme_de(self, language='deu-Latn'):
        epi = epitran.Epitran(language) # lang='deu-Latn' oder 'eng-Latn'
        tokenizer = RegexpTokenizer(r'\w+')
        
        temp_dictionary = {}
        for k, v in self.data.items():
            temp_value = []
            for line in v:
                list_of_phonemes = ''
                for word in tokenizer.tokenize(line[0].lower()):
                    try:
                        phoneme = epi.transliterate(word)
                        list_of_phonemes+= phoneme +' '
                    except:
                        list_of_phonemes+= word +' '
                line = list(line)
                line[0] = list_of_phonemes
                line = tuple(line)
                temp_value.append(line)    
            temp_dictionary[k] = temp_value
        self.data = temp_dictionary    
    
    def get_random_poems(self, n):
        temp = {}
        i = 0
        for k, v in self.data.items():
            if i < n:
                temp[k]=v
            i += 1 
        self.data = temp
                
    
    # uses cmu dict for look ups. very slow -  should only be used for a few poems
    def to_phoneme_en(self):
        tokenizer = RegexpTokenizer(r'\w+')
        
        temp_dictionary = {}
        for k, v in self.data.items():
            temp_value = []
            for line in v:
                list_of_phonemes = []
                for word in tokenizer.tokenize(line[0].lower()):
                    try:
                        phoneme = cmudict.dict()[word]
                        list_of_phonemes.append(phoneme[0])
                    except:
                        list_of_phonemes.append([word])
                line = list(line)
                line[0] = list_of_phonemes
                line = tuple(line)
                temp_value.append(line)    
            temp_dictionary[k] = temp_value
        self.data = temp_dictionary    
        
    def print_data(self):
        pprint(self.data)

    # used for Mallet
    def poems_to_single_string(self):
        poems = []
        for value in self.data.values():
            poem = ''
            for verse in value:
                poem+=verse[0]+' '
            poems.append(poem)
        return poems
    
    # create sentiments, exports results to file
    def make_sentiment_list(self, wordlist_pos, wordlist_neg, export_file):
        tokenizer = RegexpTokenizer(r'\w+')
        wordlist_pos_set = set()
        wordlist_neg_set = set()
        all_ratings = []
        #load wordlist to set
        with open(wordlist_pos, 'r') as file:
            for word in file:
                wordlist_pos_set.add(word.lower().strip())
        with open(wordlist_neg, 'r') as file:
            for word in file:
                wordlist_neg_set.add(word.lower().strip())
        print('loaded stuff, start iterating...')
        #iterate over all poems
        for k,v in self.data.items():
            counter = Counter()
            #iterate over each poem creating a counter
            for line in v:
                for word in tokenizer.tokenize(line[0]):
                    counter[word.lower()] += 1
            rating = 0
            all_found_ratings = 0
            # sum rating 
            for word in wordlist_neg_set:
                rating -= counter[word]
                all_found_ratings+= counter[word]
            for word in wordlist_pos_set:
                rating += counter[word]  
                all_found_ratings+= counter[word]
            #print(rating)
            try:
                relative_rating = rating/all_found_ratings
            except:
                relative_rating = 0
            print(v[0][3])
            all_ratings.append((k, rating, all_found_ratings, relative_rating, v[0][3]))
        
        with open(export_file, 'w') as file:
            for poem in all_ratings:
                file.write('%s ,%s, %s, %s, %s\n' %(poem[0], poem[1], poem[2], poem[3], poem[4]))
        print('done')
        
        
    def make_sentiment_list_stanza_level_from_conll_file_textgrid(self, wordlist_pos, wordlist_neg, export_file):
        def string_replace_multiple(input_string):
            bad_char = [('ä', 'ae'), ('ö',  'oe'), ('ü', 'ue'),('ß', 'ss')]
            for i in range(len(bad_char)):
                input_string = input_string.replace(bad_char[i][0], bad_char[i][1])
            return input_string
        
        #tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_thomas.ndjson')
        tokenizer = RegexpTokenizer(r'\w+')
        wordlist_pos_set = set()
        wordlist_neg_set = set()
        all_ratings = []
        #load wordlist to set
        with open(wordlist_pos, 'r') as file:
            for word in file:
                wordlist_pos_set.add(string_replace_multiple(word).lower().encode(encoding='ascii', errors='backslashreplace').decode('utf-8').strip())
        with open(wordlist_neg, 'r') as file:
            for word in file:
                wordlist_neg_set.add(string_replace_multiple(word).lower().encode(encoding='ascii', errors='backslashreplace').decode('utf-8').strip())
        print('loaded stuff, start iterating...')
        
        #load quatrains
        data_train = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/gutentag_20k/train.txt', sep='\t', usecols = (0,1), header=None, skip_blank_lines=False)
        print('hallo')
        all_quatrains = []
        quatrain = ''
        
        ########################### simpler than for chicago and deepspeare but works better
        for i in range(len(data_train)):
            if type(data_train.iat[i,1]) == str:
                if data_train.iat[i,1] == 'sos':
                    all_quatrains.append((i, quatrain))
                    quatrain='sos '
                else:
                    quatrain += data_train.iat[i,1]+' '
        all_quatrains.append((i, quatrain))
        all_quatrains = all_quatrains[1:]
        print(len(all_quatrains))
        print(all_quatrains[0])
        print(all_quatrains[1])
        ###########################
        
           
        #iterate over all poems
        for q in all_quatrains:
#             print(q)
            counter = Counter()
            #iterate over each poem creating a counter
            for word in tokenizer.tokenize(q[1]):
                counter[word.lower()] += 1
            rating = 0
            all_found_ratings = 0
            # sum rating 
            for word in wordlist_neg_set:
                rating -= counter[word]
                all_found_ratings+= counter[word]
            for word in wordlist_pos_set:
                rating += counter[word]  
                all_found_ratings+= counter[word]
            #print(rating)
            try:
                relative_rating = rating/all_found_ratings
            except:
                relative_rating = 0
            all_ratings.append((q[0], rating, all_found_ratings, relative_rating)) #q[0] is index in conll
            if len(all_ratings) % 1000 == 0:
                print(len(all_ratings))
        with open(export_file, 'w') as file:
            for poem in all_ratings:
                file.write('%s ,%s, %s, %s\n' %(poem[0], poem[1], poem[2], poem[3]))
        print('done')


        
        
        
        
    def make_sentiment_list_stanza_level_from_conll_file(self, wordlist_pos, wordlist_neg, export_file):
        def string_replace_multiple(input_string):
            bad_char = [('ä', 'ae'), ('ö',  'oe'), ('ü', 'ue'),('ß', 'ss')]
            for i in range(len(bad_char)):
                input_string = input_string.replace(bad_char[i][0], bad_char[i][1])
            return input_string
        
        #tg = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_thomas.ndjson')
        tokenizer = RegexpTokenizer(r'\w+')
        wordlist_pos_set = set()
        wordlist_neg_set = set()
        all_ratings = []
        #load wordlist to set
        with open(wordlist_pos, 'r') as file:
            for word in file:
                wordlist_pos_set.add(string_replace_multiple(word).lower().encode(encoding='ascii', errors='backslashreplace').decode('utf-8').strip())
        with open(wordlist_neg, 'r') as file:
            for word in file:
                wordlist_neg_set.add(string_replace_multiple(word).lower().encode(encoding='ascii', errors='backslashreplace').decode('utf-8').strip())
        print('loaded stuff, start iterating...')
        
        #load quatrains
        data_train = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/textgrid/train.txt', sep='\t', usecols = (0,1), header=None, skip_blank_lines=False)
        print('hallo')
        all_quatrains = []
        quatrain = ''
        for i in range(len(data_train)):
            if type(data_train.iat[i, 1]) == str:
                if data_train.iat[i,1] != 'sos':
                    if data_train.iat[i,1] == 'newline':
                        quatrain+='. '
                    else:
                        quatrain+=data_train.iat[i,1]+' '
            else:
                all_quatrains.append((data_train.iat[i- len(tokenizer.tokenize(quatrain)), 0],quatrain))
                quatrain = ''
        print(all_quatrains[0])

        
        print(len(all_quatrains))
          
        #iterate over all poems
        for q in all_quatrains:
#             print(q)
            counter = Counter()
            #iterate over each poem creating a counter
            for word in tokenizer.tokenize(q[1]):
                counter[word.lower()] += 1
            rating = 0
            all_found_ratings = 0
            # sum rating 
            for word in wordlist_neg_set:
                rating -= counter[word]
                all_found_ratings+= counter[word]
            for word in wordlist_pos_set:
                rating += counter[word]  
                all_found_ratings+= counter[word]
            #print(rating)
            try:
                relative_rating = rating/all_found_ratings
            except:
                relative_rating = 0
            all_ratings.append((q[0], rating, all_found_ratings, relative_rating)) #q[0] is index in conll
            if len(all_ratings) % 1000 == 0:
                print(len(all_ratings))
        with open(export_file, 'w') as file:
            for poem in all_ratings:
                file.write('%s ,%s, %s, %s\n' %(poem[0], poem[1], poem[2], poem[3]))
        print('done')

        
    def get_sentiment_top_bottom_n_poems(self, sentiment_file, top_bottom_n):
        data = pd.read_csv(sentiment_file, header=None, names=('key', 'rating', 'all_found_ratings', 'relative_rating', 'year'))
        data = data[data.all_found_ratings > 4]
        print(len(data))
        most_pos = data.nlargest(top_bottom_n, 'relative_rating')['key'].tolist()
        most_neg = data.nsmallest(top_bottom_n, 'relative_rating')['key'].tolist()
        print(most_pos, most_neg)

        
        
        
    # makes a plain text file with unique words. This file can be used for g2p conversion with g2p-transformer
    def export_data_for_g2p(self):
        vocabulary = set()
        tokenizer = RegexpTokenizer(r'\w+')
        for poem in self.data.values():
            for line in poem:
                for token in tokenizer.tokenize(line[0]):
                    vocabulary.add(token.lower())
        with open ('vocab.txt','w') as file:
            for token in vocabulary:
                file.write("%s , \n" %token)
        print(len(vocabulary))
    
    


    #transform given poem-values to string and calls aylien for sentiment rating. returns sentiment
    def aylien_for_sentiment(self, poem, language):
        poem_str =''
        for line in poem:
            poem_str+= line[0].lower()+' '
        client = textapi.Client('3e6d3188', 'a59809a95ed4dbf11af753e374b64605')
        sentiment = client.Sentiment({'language': language,'text': poem_str })
        return sentiment['polarity']

    def get_aylien_sentiment_on_stanza_level(self, start, end):
        #load quatrains
        data_train = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/deepspeare/stanza_w_alits_density/train_stanza_alit_density_rhyme.txt', sep='\t', usecols = (0,1), header=None, skip_blank_lines=False)
        tokenizer = RegexpTokenizer(r'\w+')
        all_quatrains = []
        quatrain = ''
        for i in range(len(data_train)):
            if type(data_train.iat[i, 1]) == str:
                if data_train.iat[i,1] != 'sos':
                    if data_train.iat[i,1] == 'newline':
                        quatrain+='. '
                    else:
                        quatrain+=data_train.iat[i,1]+' '
            else:
                all_quatrains.append((data_train.iat[i- len(tokenizer.tokenize(quatrain)), 0],quatrain))
                quatrain = ''
        
        client = textapi.Client('3e6d3188', 'a59809a95ed4dbf11af753e374b64605')
        
        aylien_rating_list =[]
        for i in range(start, end):
            sentiment = client.Sentiment({'language': 'en','text': all_quatrains[i][1] })
            aylien_rating_list.append((all_quatrains[i][0], sentiment['polarity']))
        print(aylien_rating_list)
        # load sentiment rating file with word lookpu ratings and add aylien ratings in separate column
        sentiment_from_lookup = pd.read_csv('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/deepspeare_stanza_sentiment_w_aylien', sep='\t', header=None, skip_blank_lines=False)
        #sentiment_from_lookup[4] = ''
        
        for value in aylien_rating_list:
            for i in range(len(sentiment_from_lookup)):
                if sentiment_from_lookup.iat[i,0] == value[0]:
                    sentiment_from_lookup.iat[i,4] = value[1] 
                    break
                
        sentiment_from_lookup.to_csv('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/deepspeare_stanza_sentiment_w_aylien', sep='\t', header=None, index=False)
    
#     def compare_sentiment_stanza_wordlist_aylien(self):
#         sentiment_from_lookup = pd.read_csv('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/deepspeare_stanza_sentiment_w_aylien', sep='\t', header=None, skip_blank_lines=False)
#         match = 0
#         no_match = 0
#         for i in range(0, 960):
#             wordlist_rating = sentiment_from_lookup.iat[i,3]
#             aylien_rating = sentiment_from_lookup.iat[i,4]
#             if (aylien_rating == 'positive' and wordlist_rating >= 0.25) or (aylien_rating == 'negative' and wordlist_rating <= -0.25) or (aylien_rating == 'neutral' and (-0.25 < wordlist_rating < 0.25)):
#                 match += 1
#             else:
# #                 if aylien_rating != 'neutral':
#                 no_match += 1
#         print(match, no_match)

    def compare_sentiment_stanza_wordlist_aylien(self):
        sentiment_from_lookup = pd.read_csv('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/deepspeare_stanza_sentiment_w_aylien', sep='\t', header=None, skip_blank_lines=False)
        pos_match = 0
        pos_all = 0
        pos_wordlist = 0
        neutral_match = 0
        neutral_all = 0
        neutral_wordlist = 0
        neg_match = 0
        neg_all = 0
        neg_wordlist = 0
        border_pos = 0.25
        border_neg = -0.25
        
        wordlist_vector = []
        aylien_vector = []
        
        for i in range(0, 1000):
            wordlist_rating = sentiment_from_lookup.iat[i,3]
            wordlist_vector.append(wordlist_rating)
            aylien_rating = sentiment_from_lookup.iat[i,4]
#             if wordlist_rating >border_pos:
#                 wordlist_vector.append(1)
#             elif wordlist_rating <border_neg:
#                 wordlist_vector.append(-1)
#             elif border_pos > wordlist_rating > border_neg:
#                 wordlist_vector.append(0)
#             else:
#                 wordlist_vector.append(0)
            
            if aylien_rating == 'positive':
                aylien_vector.append(1)
            elif aylien_rating  == 'negative':
                aylien_vector.append(-1)
            elif aylien_rating == 'neutral':
                aylien_vector.append(0)
        print(len(wordlist_vector), len(aylien_vector))
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import itertools
        from scipy.stats import spearmanr
        print('spearman: ', spearmanr(wordlist_vector, aylien_vector))
        cnf_mat = confusion_matrix(wordlist_vector, aylien_vector, [1, 0, -1])
        
        
        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, Sentiment Comparison \n 1000 Quatrains')
        
            print(cm)
        
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
        
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
            plt.xlabel('Aylien label')
            plt.ylabel('Wordlist label')
            plt.tight_layout()
        plt.figure()
        plot_confusion_matrix(cnf_mat, classes=['pos', 'neutral', 'neg'], title='Confusion matrix, Sentiment Comparison \n on 1000 Quatrains')
        plt.show()
        
        
#         same = 0
#         one_off = 0
#         two_off = 0
#         for w,a in zip(wordlist_vector, aylien_vector):
#             if w == a:
#                 same += 1
#             elif abs(w-a) == 1:
#                 one_off += 1
#             elif abs(w-a) == 2:
#                 two_off += 1
#         print(same, one_off, two_off)
        
#             if aylien_rating == 'positive':
#                 pos_all += 1
#                 if wordlist_rating >= border_pos:
#                     pos_match += 1
#             if aylien_rating == 'negative':
#                 neg_all += 1
#                 if wordlist_rating <= border_neg:
#                     neg_match += 1
#             if aylien_rating == 'neutral':
#                 neutral_all += 1
#                 if border_pos > wordlist_rating > border_neg:
#                     neutral_match += 1
#             if wordlist_rating >= border_pos:
#                 pos_wordlist += 1
#             elif wordlist_rating <= border_neg:
#                 neg_wordlist += 1
#             elif border_pos > wordlist_rating > border_neg:
#                 neutral_wordlist += 1 
#         print('aylien ratings', 'agree', 'wordlist rating')
#         print(pos_all, pos_match, pos_wordlist)
#         print(neutral_all, neutral_match, neutral_wordlist)
#         print(neg_all, neg_match, neg_wordlist)
#         print('Gleich bewertet: ', pos_match+neg_match+neutral_match, 'prozentual: ', (pos_match+neg_match+neutral_match) /1000)
        
    # reads in sentiment file, finds poems with top and bottom n sentiment
    def aylien_test_and_export_sentiment_ranking_on_top_n_poems(self, sentiment_file, top_bottom_n, language):
        data = pd.read_csv(sentiment_file, header=None, names=('key', 'rating', 'all_found_ratings', 'relative_rating'))#, 'year'))
        data = data[data.all_found_ratings > 4]
        print(len(data))
        most_pos = data.nlargest(top_bottom_n, 'relative_rating')['key'].tolist()
        most_neg = data.nsmallest(top_bottom_n, 'relative_rating')['key'].tolist()
        print(most_pos, most_neg)
#         aylien_rating_pos = []
#         aylien_rating_neg = []
#         for poem in most_pos:
#             sent = textgrid.aylien_for_sentiment(textgrid.data[str(poem)], language)
#             aylien_rating_pos.append(sent)#, 'label: pos')
#         for poem in most_neg:
#             sent = textgrid.aylien_for_sentiment(textgrid.data[str(poem)], language)
#             aylien_rating_neg.append(sent)#, 'label: neg')
#         aylien_rating_pos.extend(aylien_rating_neg)
#         most_pos.extend(most_neg)
#         with open('aylien_ratings_textgrid_top30_low30.txt', 'w') as file:
#             for i in range(len(aylien_rating_pos)):
#                 file.write('%s , %s \n' %(most_pos[i], aylien_rating_pos[i]))
#         print('done')

    def aylien_test_and_export_sentiment_ranking_on_random_n_poems(self, sentiment_file, random_n, language):
        keys_random = list(self.data.keys())
        keys_random = random.sample(keys_random, random_n)
        aylien_ratings = []
        for key in keys_random:
            sent = self.aylien_for_sentiment(self.data[str(key)], language)
            aylien_ratings.append(sent)
        wordlist_ratings = []
        data = pd.read_csv(sentiment_file, header=None, names=('key', 'rating', 'all_found_ratings', 'relative_rating', 'year'))
        wordlist_rating_number = []
        for key in keys_random:
            rating = list(data[data.key == int(key)]['relative_rating'])[-1]
            wordlist_rating_number.append(rating)
            if rating > 0.33:
                rating = 'positive'
            elif rating < -0.33:
                rating = 'negative'
            else:
                rating = 'neutral'
            wordlist_ratings.append(rating)
        counter_aylien  = Counter()
        counter_wordlist = Counter()
         
        with open('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/chicago_aylien_wordlist_comp_on_random_stanza_50.txt', 'w') as file:
            for i in range(len(keys_random)):
                if i == 0:
                    file.write('key, aylien, wordlist, wordlist_number \n')
                file.write('%s,  %s, %s, %s \n' %(keys_random[i], aylien_ratings[i], wordlist_ratings[i], wordlist_rating_number[i]))
            
        
        
    # this method can only be used for one string. otherwise g2p tool crashes
    def alliteration_in_line(self, g2p_file, verse):
#         g2p_lookup = {}
#         with open(g2p_file, 'r') as file:
#             for line in file:
#                 temp = line.split(" ", 1)
#                 g2p_lookup[temp[0]] = temp[1]
        
        def window(seq, n):
            it = iter(seq)
            result = tuple(islice(it, n))
            if len(result) == n:
                yield result
            for element in it:
                result = result[1:] + (element,)
                yield result
        
        verse = g2p_seq2seq.app.main([verse.lower()])
        print('verse: ', verse)
#         verse = [g2p_lookup[token] for token in verse] #if token not in stopwords.words('english')]
        
        twograms = window(verse[0], 2)
        threegrams = window(verse[0], 3)
        alits_two = []
        alits_three = []
        for gram in twograms:
            print(gram)
            if gram[0][0] == gram[1][0]:
                alits_two.append(gram)
        for gram in threegrams:
            if gram[0][0] == gram[1][0] == gram[2][0]:
                alits_three.append(gram)        
        temp = []
        for twog in alits_two:
            for threeg in alits_three:
                if not set(twog).issubset(threeg):
                    temp.append(twog)
        alits_two = temp
        print(alits_three, alits_two)
        #useless because it just counts pos, neg and neutral but doesen connect it with poem id... 
#         for i in range(len(wordlist_ratings)):
#             counter_aylien[aylien_ratings[i]] += 1
#             counter_wordlist[wordlist_ratings[i]] += 1
#         print('Aylien', counter_aylien)
#         print('Wordlist', counter_wordlist)


    #read in file that has been translated to phonemes in CONLL shape: word W OH RD 
    #counts bi- and trigram alliterations
    def read_g2p_converted_file_check_for_aliterations(self):
        def window(seq, n):
            it = iter(seq)
            result = tuple(islice(it, n))
            if len(result) == n:
                yield result
            for element in it:
                result = result[1:] + (element,)
                yield result
                
        data = pd.read_csv('/home/joerg/workspace/g2p_raw/g2p-seq2seq/chicago_aliteration_converted', sep=' ', usecols = (0,1), header=None)
        #data = pd.read_csv('/home/joerg/workspace/g2p_raw/g2p-seq2seq/deepspeare_aliteration_converted', sep=' ', usecols = (0,1), header=None)
        p = data[1].tolist()
        temp = []
        all_verses = []
        for token in p:
            if type(token) is float: #float is NaN and this is marks a new verse in the data
                all_verses.append(temp)
                temp = []
            else:
                temp.append(token)
        
        twograms = []
        for verse in all_verses:
            twograms.append(window(verse, 2))
        threegrams  = []
        for verse in all_verses:
            threegrams.append(window(verse, 3))
            
        alits_two = []
        for gram in twograms:
            for g in gram:
                if g[0] == g[1]:
                    alits_two.append(g)
        alits_three = []
        for gram in threegrams:
            for g in gram:
                if g[0] == g[1] == g[2]:
                    alits_three.append(g)
        
        print('aliteration bigrams ', len(alits_two), ' aliterations trigram ',len(alits_three), ' Anzahl Verse ', len(all_verses))
        #print(alits_two, alits_three)
    
    
    def plot_period_of_publication(self):
        release_counter = Counter()
        for k, v in self.data.items():
            temp_year = v[0][3][:-2]
            if len(temp_year) == 2:
                release_counter[temp_year] += 1
    
        print(sorted(release_counter.items()))
        
        labels, values = zip(*sorted(release_counter.items()))
        labels = [label+'00' for label in labels]
        print(labels)
        indexes = np.arange(len(labels))
        width = 0.75
        plt.bar(indexes, values, align='center')
        plt.xticks(indexes, labels)
        plt.xlabel('Period')
        plt.ylabel('Number of Poems')
        plt.title('Textgrid Corpus, Period of Publication')
        plt.show()
###########
# poems are stored in textgrid.data as a dictionary with poem_no as key and list of tuples as values
# dictionary values: list of tuples. Tuple shape is like this: (verse, rhyme annotation, stanza number, release date, author)
###########

if __name__ == '__main__':
    
    corpus = Corpus('/home/joerg/workspace/thesis/Deepspeare_Data/deepspeare_data.ndjson')
    for x in corpus.data.values():
        for verse in x:
            print(verse[0])


#     corpus.filter_by_year(1700, 2000)
#     corpus.filter_by_no_of_lines(8, 100)
#     print(len(corpus.data))
    
#     for poem in corpus.data.values():
#         temp = []
#         
#         for line in poems:
#             if line[2]
# 
#     print(len(corpus.data))
    
#     with open('gutentag_20k_poems', 'w') as file:
#         for k, poem in corpus.data.items():
#             for line in poem:
#                 file.write('{"s": '+ '"'+line[0].strip()+'", '+ '"rhyme": ' + '"' + '__'+'", ' + '"poem_no": ' + '"' + str(k)+'", ' + '"stanza_no": '+ '"'+str(line[2])+ '", '+ '"released": ' + '"' + str(line[3])+'"' + '"author": ' + '"__"' +'}\n')
#     corpus = corpus.filter_by_no_of_lines(8, 100)
    
#     
#     for poem in corpus.data.values():
#         for verse in poem:
#             if len(verse[3]) != 4:
#                 print(verse[3])
#             
            

#     corpus.filter_by_year(1200, 1400)
    
#     corpus.make_sentiment_list_stanza_level_from_conll_file_textgrid('/home/joerg/workspace/thesis/Sentiment/en/opinion-lexicon-English/positive-words.txt', '/home/joerg/workspace/thesis/Sentiment/en/opinion-lexicon-English/negative-words.txt', '/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/gutentag_20k/tttrrrraain')



#     corpus.get_sentiment_top_bottom_n_poems('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/chicago_stanza_sentiment_train', 10)
    
#     corpus.compare_sentiment_stanza_wordlist_aylien()

#     corpus.plot_period_of_publication()

    
#     chicago.compare_sentiment_stanza_wordlist_aylien()

#     for start in [1080, 1110]:
#         chicago.get_aylien_sentiment_on_stanza_level(start, start+30)
#         time.sleep(3)
    
#     chicago.compare_sentiment_stanza_wordlist_aylien()

#     chicago.make_sentiment_list_stanza_level_from_conll_file('/home/joerg/workspace/thesis/Sentiment/en/opinion-lexicon-English/positive-words.txt', '/home/joerg/workspace/thesis/Sentiment/en/opinion-lexicon-English/negative-words.txt', 'chicago_stanza_sentiment_dev')
#     chicago.aylien_test_and_export_sentiment_ranking_on_random_n_poems('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/chicago_stanza_sentiment', 50, 'en')
    
    
#     ds = Corpus('/home/joerg/workspace/thesis/Deepspeare_Data/deepspeare_data.ndjson')
# #     ds.get_sentiment_top_bottom_n_poems('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/sentiment_deepspeare.txt', 20, 'en')
#     ds_most_pos = [3216, 1845, 1023, 2485, 1416, 2342, 3046, 746, 2983, 3245, 2530, 2928, 846, 2330, 720, 2006, 2130, 81, 2961, 1387]
#     ds_most_neg = [214, 1867, 2835, 2932, 1744, 787, 163, 240, 390, 2327, 1844, 3202, 353, 2676, 3047, 2419, 2198, 820, 426, 312]
#     for poem in ds_most_neg:
#         poem_string = ''
#         for values in ds.data[str(poem)]:
#             poem_string+=values[0]
#         print(poem_string)

#     textgrid.make_sentiment_list('/home/joerg/workspace/thesis/Sentiment/de/GermanPolarityClues-2012/GermanPolarityClues-Positive-21042012.tsv', '/home/joerg/workspace/thesis/Sentiment/de/GermanPolarityClues-2012/GermanPolarityClues-Negative-21042012.tsv', 'textgrid_thomas_sentiment')
        
#     textgrid = Corpus('/home/joerg/workspace/thesis/Chicago/chicago.ndjson')
#     textgrid.read_g2p_converted_file_check_for_aliterations()
#     textgrid.alliteration_in_line('/home/joerg/workspace/thesis/Interface/g2p/g2p_chicago.txt', "My Muse made enfranchised enfranchised")

    
