'''
Created on Nov 20, 2018

@author: joerg
'''
import ndjson
import pprint 
import pyphen
import epitran
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import RegexpTokenizer

class Poem:
    def __init__(self):
        self.date = 0
        self.poem_no = 0
        self.stanza = [] # list of lists of tuples (verse, rhyme)
        
class Corpus:
    def __init__(self, filename):
        self.data = self.__load_data(filename)
        self.__poems = []
                
    def __load_data(self, filename):
        with open(filename, 'rb') as file:
            data = ndjson.load(file)
        print('Data loaded')
        return data
    
    def __create_poems(self):
        poem_ids = set()
        for line in self.data:
            poem_ids.add(line['poem_no']) #get unique number of poem
        
        for id in sorted(list(poem_ids)): # für alle poems in self.data
            p = Poem()
            poem_temp  = [] # lines die zu einem Poem gehören
            for line in self.data:
                if int(line['poem_no']) == int(id):
                    poem_temp.append(line)
            for i in range(1, int(poem_temp[-1]['stanza_no'])+1): # p.stanza = list(alle stanza) of list(stanza) of tuples(verse, rhyme)
                stanza = [(line['s'], line['rhyme']) for line in poem_temp if line['stanza_no'] == str(i)]
                p.stanza.append(stanza)
            p.date = int(poem_temp[0]['released'])
            p.poem_no = int(poem_temp[0]['poem_no'])
            self.__poems.append(p)

            
    def filter_by_release(self, released_from, released_to):
        self.data = [line for line in self.data if int(line['released']) >= released_from and int(line['released']) < released_to]
        self.get_number_of_poems()
    
    def filter_by_verse_length(self, max_length):
        tokenizer = RegexpTokenizer(r'\w+')
        current = int(self.data[0]['poem_no'])
        token_per_poem = []
        line_count = 0
        filtered_data = []
        poem_temp = []
        for line in self.data:
            if int(line['poem_no']) == current:
                poem_temp.append(line)
                line_count += 1
                token_per_poem.extend(tokenizer.tokenize(line['s']))
            else:
                if (len(token_per_poem) / line_count) < max_length:
                    filtered_data.extend(poem_temp)
                current = int(line['poem_no'])
                poem_temp=[]
                poem_temp.append(line)
                line_count = 1
                token_per_poem = []
                token_per_poem.extend(tokenizer.tokenize(line['s']))
        self.data = filtered_data
        self.get_number_of_poems()
    
    def filter_by_author(self, author):
        self.data = [line for line in self.data if line['author'] == author]
        self.get_number_of_poems()
    
    def filter_by_sonnett(self):
        def filter_by_verse_length_sonnet(min_length, max_length):
            tokenizer = RegexpTokenizer(r'\w+')
            current = int(self.data[0]['poem_no'])
            token_per_poem = []
            line_count = 0
            filtered_data = []
            poem_temp = []
            for line in self.data:
                if int(line['poem_no']) == current:
                    poem_temp.append(line)
                    line_count += 1
                    token_per_poem.extend(tokenizer.tokenize(line['s']))
                else:
                    if (len(token_per_poem) / line_count) < max_length and (len(token_per_poem) / line_count) > min_length:
                        filtered_data.extend(poem_temp)
                    current = int(line['poem_no'])
                    poem_temp=[]
                    poem_temp.append(line)
                    line_count = 1
                    token_per_poem = []
                    token_per_poem.extend(tokenizer.tokenize(line['s']))
            self.data = filtered_data
            self.get_number_of_poems()

        filter_by_verse_length_sonnet(7, 12)
        
        poem_ids = set()
        
        for line in self.data:
            poem_ids.add(line['poem_no'])
        print(poem_ids)
        for id in sorted(list(poem_ids)): # für alle poems in self.data
            p = Poem()
            poem_temp  = [] # alle lines die zu einem Poem gehören
            for line in self.data:
                if int(line['poem_no']) == int(id):
                    poem_temp.append(line)
            if len(poem_temp) == 14 and poem_temp[-1]['stanza_no'] == '4':
                for i in range(1, 5): # p.stanza = list(alle stanza) of list(stanza) of tuples(verse, rhyme)
                    stanza = [(line['s'], line['rhyme']) for line in poem_temp if line['stanza_no'] == str(i)]
                    p.stanza.append(stanza)
                    p.date = int(poem_temp[0]['released'])
                    p.poem_no = int(poem_temp[0]['poem_no'])
                    self.__poems.append(p)

    def get_sonnett(self, sonnett_no):
        try:
            print('Sonnet at index ', sonnett_no, 'Released: ', self.__poems[sonnett_no].date, 'Poem_no: ', self.__poems[sonnett_no].poem_no)
            pprint.pprint(self.__poems[sonnett_no].stanza)
        except Exception as e:
            print(e)
        
    def get_sentiment(self, language, poem_no): # Input datatype: String 
        tokenizer = RegexpTokenizer(r'\w+')
        pos_words = set()
        neg_words = set()
        
        if language == 'de':
            with open('/home/joerg/workspace/thesis/Sentiment/de/SentiWS_v2/SentiWS_v2_Positive.txt', 'r') as file:
                for line in file:
                    token = tokenizer.tokenize(line)
                    for word in token:
                        pos_words.add(word)
            
            with open('/home/joerg/workspace/thesis/Sentiment/de/SentiWS_v2/SentiWS_v2_Negative.txt', 'r') as file:
                for line in file:
                    token = tokenizer.tokenize(line)
                    for word in token:
                        neg_words.add(word)
        elif language =='en':
            with open('/home/joerg/workspace/thesis/Sentiment/en/opinion-lexicon-English/negative-words.txt', 'r') as file:
                for line in file:
                    token = tokenizer.tokenize(line)
                    for word in token:
                        pos_words.add(word)
            
            with open('/home/joerg/workspace/thesis/Sentiment/en/opinion-lexicon-English/positive-words.txt', 'r') as file:
                for line in file:
                    token = tokenizer.tokenize(line)
                    for word in token:
                        neg_words.add(word)
            
            
            
        poem = [line['s'] for line in self.data if line['poem_no'] == poem_no]
        
        token_in_poem = [token for line in poem for token in tokenizer.tokenize(line)]
        sentiment = 0
        found = 0
        for token in token_in_poem: 
            if token in pos_words:
                sentiment += 1
                found += 1
        for token in token_in_poem: 
            if token in neg_words:
                sentiment -= 1
                found += 1
        #print('sentiment: ', sentiment, 'found words: ', found)
        return poem_no+','+ str(sentiment)
        
    def syllablificate(self, language='en_EN'):
        language = pyphen.Pyphen(lang=language) # lang='de_DE' oder 'en_EN'
        for line in self.data:
            temp = ''
            for word in line['s'].split():
                temp += language.inserted(word+' ')
            line['s'] = temp.strip()
            
    def to_phoneme_de(self, language='deu-Latn'):
        epi = epitran.Epitran(language) # lang='deu-Latn' oder 'eng-Latn'
        for line in self.data:
            temp = ''
            for word in line['s'].split():
                #print(word)
                try:
                    phoneme = epi.transliterate(word)
                    print(phoneme)
                    temp = temp + phoneme+' '
                    #print(type(phoneme), phoneme, word)
                except:
                    temp += word.join(' ')
            #print(temp)
            line['s'] = temp.strip()
    
    def to_phoneme_en(self):
        tokenizer = RegexpTokenizer(r'\w+')
        for line in self.data:
            temp = []
            for word in tokenizer.tokenize(line['s'].lower()):
                try:
                    temp.append(cmudict.dict()[word][0])
                except:
                    temp.append([word])
            line['s'] = temp
            print(temp)
    
    def get_poem(self, poem_no):
        self.__poems = []
        self.__create_poems()
        print('Number of  Poems: ', len(self.__poems))
        try:
            print('Poem at index ', poem_no, 'Released: ', self.__poems[poem_no].date, 'Poem_no: ', self.__poems[poem_no].poem_no)
            pprint.pprint(self.__poems[poem_no].stanza)
            return self.__poems[poem_no]
        except Exception as e:
            print(e)
    
    def get_number_of_poems(self):
        nop = set()
        for line in self.data:
            nop.add(int(line['poem_no']))
        print('Number of poems: ',len(nop))
        
#     def __extract_stanzas(self, poem_no):
#         stanzas = []
#         poem = [line for line in self.raw_data if line['poem_no'] == str(poem_no)]
#         date = poem['release_year']
#         for i in range(int(poem[-1]['stanza_no'])):
#             stanza = [line for line in poem if line['stanza_no'] == str(i)]
#             stanzas.append(stanza)
#         return stanzas, str(date)

    def get_all_sonnets(self):
        def filter_by_verse_length_sonnet(min_length, max_length):
            tokenizer = RegexpTokenizer(r'\w+')
            current = int(self.data[0]['poem_no'])
            token_per_poem = []
            line_count = 0
            filtered_data = []
            poem_temp = []
            for line in self.data:
                if int(line['poem_no']) == current:
                    poem_temp.append(line)
                    line_count += 1
                    token_per_poem.extend(tokenizer.tokenize(line['s']))
                else:
                    if (len(token_per_poem) / line_count) < max_length and (len(token_per_poem) / line_count) > min_length:
                        filtered_data.extend(poem_temp)
                    current = int(line['poem_no'])
                    poem_temp=[]
                    poem_temp.append(line)
                    line_count = 1
                    token_per_poem = []
                    token_per_poem.extend(tokenizer.tokenize(line['s']))
            self.data = filtered_data
            self.get_number_of_poems()

        filter_by_verse_length_sonnet(7, 12)
        
        poem_ids = set()
        
        for line in self.data:
            poem_ids.add(line['poem_no'])
        print(poem_ids) # Poem_nos aller Sonnette
        poem_dict ={}
        for id in poem_ids:
            poem_dict[id] = []

        for line in self.data:
            if line['poem_no'] in poem_ids:
                poem_dict[line['poem_no']].append(line['s'])
        print(poem_dict['31020'])
    # noch weitere Filter anwenden für Anzahl an verses
    # :) export poem_dict! 


def get_dta_data_for_embeddings():
    dta = Corpus('/home/joerg/workspace/thesis/German_Annotated/dta_annotated.ndjson')
    dta.filter_by_verse_length(12)
    data4emb = []
    for v in dta.data:
        data4emb.append(nltk.word_tokenize((v['s'].lower())))
    return data4emb

def dta_for_deepspeare():
    dta = Corpus('/home/joerg/workspace/thesis/German_Annotated/dta_annotated.ndjson')
    dta.filter_by_verse_length(12)
    dta.get_poem(1826)
    dta.get_poem(1827)
    dta.get_poem(1829)
    train_data= []
    for i in range(1,1831):
        poem = ''
        for line in dta.data:
            if line['poem_no'] == str(i):
                poem += line['s'].strip()+ ' <eos> '
        train_data.append(poem)
    with open('deep_de.txt', 'w') as file:
        for poem in train_data:
            file.write('%s\n'%poem)
        
        
if __name__ == '__main__':
    #dta_for_deepspeare()
    textgrid = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups_p1.ndjson')
    textgrid.to_phoneme_de()
    #textgrid.get_all_sonnets()
    
    
    
    # example usage:
#     chicago = Corpus('/home/joerg/workspace/thesis/Chicago/chicago.ndjson')
#     chicago.filter_by_release(1700, 1800)
#     chicago.get_poem(16)

    # exporting sentiments:
    sentiment_lookup = []
    '''
    textgrid = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups_p1.ndjson')
    print('textgrid')
    for i in range(1, int(textgrid.data[-1]['poem_no'])):
        sentiment_lookup.append(textgrid.get_sentiment('de', str(i)))
    with open('/home/joerg/workspace/thesis/Textgrid/textgrid_sentiment_p1.txt', 'w') as file:
        for line in sentiment_lookup:
            file.write('%s\n' % line)
    textgrid = Corpus('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re__nodups_p2.ndjson')
    print('textgrid')
    for i in range(1, int(textgrid.data[-1]['poem_no'])):
        sentiment_lookup.append(textgrid.get_sentiment('de', str(i)))
    with open('/home/joerg/workspace/thesis/Textgrid/textgrid_sentiment_p2.txt', 'w') as file:
        for line in sentiment_lookup:
            file.write('%s\n' % line)
    '''
    '''

    gt = Corpus('/home/joerg/workspace/thesis/Gutentag/gutentag_en_all_lines_from_single_file_p1.ndjson')
    print('gt')
    for i in range(1, int(gt.data[-1]['poem_no'])):
        sentiment_lookup.append(gt.get_sentiment('en', str(i)))
    with open('/home/joerg/workspace/thesis/Gutentag/gutentag_en_sentiment_p1.txt', 'w') as file:
        for line in sentiment_lookup:
            file.write('%s\n' % line)
    '''
    '''
            
    gt = Corpus('/home/joerg/workspace/thesis/Gutentag/gutentag_en_all_lines_from_single_file_p2.ndjson')
    print('gt')
    for i in range(1, int(gt.data[-1]['poem_no'])):
        sentiment_lookup.append(gt.get_sentiment('en', str(i)))
    with open('/home/joerg/workspace/thesis/Gutentag/gutentag_en_sentiment_p2.txt', 'w') as file:
        for line in sentiment_lookup:
            file.write('%s\n' % line)
    dta = Corpus('/home/joerg/workspace/thesis/German_Annotated/dta_annotated.ndjson')
    print('dta')
    for i in range(1, int(dta.data[-1]['poem_no'])):
        sentiment_lookup.append(dta.get_sentiment('de', str(i)))
    with open('/home/joerg/workspace/thesis/German_Annotated/dta_sentiment.txt', 'w') as file:
        for line in sentiment_lookup:
            file.write('%s\n' % line)
      
    ds = Corpus('/home/joerg/workspace/thesis/Deepspeare_Data/deepspeare_data.ndjson')
    print('ds')
    for i in range(1, int(ds.data[-1]['poem_no'])):
        sentiment_lookup.append(ds.get_sentiment('en', str(i)))
    with open('/home/joerg/workspace/thesis/Deepspeare_Data/deepspeare_sentiment.ndjson', 'w') as file:
        for line in sentiment_lookup:
            file.write('%s\n' % line)
              
  
    chicago = Corpus('/home/joerg/workspace/thesis/Chicago/chicago.ndjson')
    print('chicago')
    for i in range(1, int(chicago.data[-1]['poem_no'])):
        sentiment_lookup.append(chicago.get_sentiment('en', str(i)))
    with open('/home/joerg/workspace/thesis/Chicago/chicago_sentiment.txt', 'w') as file:
        for line in sentiment_lookup:
            file.write('%s\n' % line)
    '''
    