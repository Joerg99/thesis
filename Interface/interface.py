'''
Created on Nov 15, 2018

@author: joerg
'''

import ndjson
import pyphen


class Reader():
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_data()
    
    
    def load_data(self):
        with open(self.filename, 'r') as file:
            data = ndjson.load(file)
        return data

    def syllablificate(self):
        language = pyphen.Pyphen(lang='en_EN') # lang='de_DE'
        data_syllables = self.data.copy()
        for i  in range(len(self.data)):
            temp_line = ''
            for word in self.data[i]['s'].split():
                temp_line += language.inserted(word+' ')
            data_syllables[i]['s'] = temp_line.strip()
        return data_syllables


if __name__ == '__main__':
    r = Reader('/home/joerg/workspace/thesis/Deepspeare_Data/deepspeare_data.ndjson')
    sylls = r.syllablificate()
    for blah in sylls[:20]:
        print(blah)