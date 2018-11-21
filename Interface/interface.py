'''
Created on Nov 20, 2018

@author: joerg
'''
import ndjson
import pprint 
import pyphen
import nltk
from nltk.tokenize import RegexpTokenizer

class Poem:
    def __init__(self):
        self.date = 0
        self.poem_no = 0
        self.stanza = []
        
class Corpus:
    def __init__(self, filename):
        self.data = self.__load_data(filename)
        self.__poems = []
                
    def __load_data(self, filename):
        with open(filename, 'rb') as file:
            data = ndjson.load(file)
        return data
    
    def __create_poems(self):
        poem_ids = set()
        for line in self.data:
            poem_ids.add(line['poem_no'])
        
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
        
    def syllablificate(self, language='en_EN'):
        language = pyphen.Pyphen(lang=language) # lang='de_DE' oder 'en_EN'
        for line in self.data:
            temp = ''
            for word in line['s'].split():
                temp += language.inserted(word+' ')
            line['s'] = temp.strip()
            
    def get_poem(self, poem_no):
        self.__create_poems()
        print('länge poems: ', len(self.__poems))
        print('Released: ', self.__poems[poem_no].date, 'Poem_no: ', self.__poems[poem_no].poem_no)
        pprint.pprint(self.__poems[poem_no].stanza)
        return self.__poems[poem_no]


#     def __extract_stanzas(self, poem_no):
#         stanzas = []
#         poem = [line for line in self.raw_data if line['poem_no'] == str(poem_no)]
#         date = poem['release_year']
#         for i in range(int(poem[-1]['stanza_no'])):
#             stanza = [line for line in poem if line['stanza_no'] == str(i)]
#             stanzas.append(stanza)
#         return stanzas, str(date)
    

if __name__ == '__main__':
    dta = Corpus('/home/joerg/workspace/thesis/German_Annotated/dta_annotated.ndjson')
    #dta.filter_by_release(1800, 1810)
    #dta.syllablificate('de_DE')
    #dta.get_poem(1)
    dta.filter_by_verse_length(5)
    dta.get_poem(360)
    