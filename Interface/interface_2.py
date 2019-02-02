'''
Created on Dec 6, 2018

@author: joerg
'''
import ndjson
from pprint import pprint
from nltk.tokenize import RegexpTokenizer
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
sys.path.insert(0, '/home/joerg/workspace/g2p-seq2seq/g2p_seq2seq')

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
        
    def get_sentiment_top_bottom_n_poems(self, sentiment_file, top_bottom_n, language):
        data = pd.read_csv(sentiment_file, header=None, names=('key', 'rating', 'all_found_ratings', 'relative_rating', 'year'))
        data = data[data.all_found_ratings > 20]
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

    # reads in sentiment file, finds poems with top and bottom n sentiment
    def aylien_test_and_export_sentiment_ranking_on_top_n_poems(self, sentiment_file, top_bottom_n, language):
        data = pd.read_csv(sentiment_file, header=None, names=('key', 'rating', 'all_found_ratings', 'relative_rating', 'year'))
        data = data[data.all_found_ratings > 20]
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
         
        with open('aylien_wordlist_comp_on_random_n.txt', 'w') as file:
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
    


###########
# poems are stored in textgrid.data as a dictionary with poem_no as key and list of tuples as values
# dictionary values: list of tuples. Tuple shape is like this: (verse, rhyme annotation, stanza number, release date, author)
###########

if __name__ == '__main__':
    
#     textgrid = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_thomas.ndjson')
#     textgrid = Corpus('/home/joerg/workspace/thesis/Chicago/chicago.ndjson')
    ds = Corpus('/home/joerg/workspace/thesis/Deepspeare_Data/deepspeare_data.ndjson')
#     ds.get_sentiment_top_bottom_n_poems('/home/joerg/workspace/thesis/Interface/sentiment_with_averages/sentiment_deepspeare.txt', 20, 'en')
    ds_most_pos = [3216, 1845, 1023, 2485, 1416, 2342, 3046, 746, 2983, 3245, 2530, 2928, 846, 2330, 720, 2006, 2130, 81, 2961, 1387]
    ds_most_neg = [214, 1867, 2835, 2932, 1744, 787, 163, 240, 390, 2327, 1844, 3202, 353, 2676, 3047, 2419, 2198, 820, 426, 312]
    for poem in ds_most_neg:
        poem_string = ''
        for values in ds.data[str(poem)]:
            poem_string+=values[0]
        print(poem_string)

#     textgrid.make_sentiment_list('/home/joerg/workspace/thesis/Sentiment/de/GermanPolarityClues-2012/GermanPolarityClues-Positive-21042012.tsv', '/home/joerg/workspace/thesis/Sentiment/de/GermanPolarityClues-2012/GermanPolarityClues-Negative-21042012.tsv', 'textgrid_thomas_sentiment')
        
#     textgrid = Corpus('/home/joerg/workspace/thesis/Chicago/chicago.ndjson')
#     textgrid.read_g2p_converted_file_check_for_aliterations()
#     textgrid.alliteration_in_line('/home/joerg/workspace/thesis/Interface/g2p/g2p_chicago.txt', "My Muse made enfranchised enfranchised")

    
