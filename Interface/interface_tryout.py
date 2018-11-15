'''
Created on 27 Oct 2018

@author: Jorg
'''
import csv
import re
import os
import pyphen
import nltk
import epitran
import pandas as pd


class Blah ():
    def __init__(self):
        self.data, self.data_len = self.load_data()
        
    def load_data(self, filename='/Users/Jorg/Documents/workspace/workspace_oxygen/thesis/XMLParser/all_german_exported.csv', language='de'):
        try:
            data = pd.read_csv(filename)
            print('Data loaded. # of examples:', data.shape[0])
        except Exception as e:
            print('Data not loaded. Error type: ', str(e))
        return data, data.shape[0]
    
    def random_select(self, n):
        texts  =[]
        for i in range(n):
            texts.append(self.data.iloc[i]['Text'])
        return texts

    def make_n_grams(self, n, sequence):
        n_grams = []
        for entry in sequence:
            n_grams.append(nltk.ngrams(entry.split(), n)) #padding????
        return n_grams
    
    def syllabificate(self, sequence):
        language = pyphen.Pyphen(lang='de_DE')
        print(language.inserted('Zeitschrift'))
    
    def word_to_phoneme(self, word):
        epi = epitran.Epitran('deu-Latn')
        print(epi.transliterate(word))


b = Blah()

selection = b.random_select(5)
grams = b.make_n_grams(3, selection)
b.syllabificate("Wahnsinn was für übertriebene Text.")
