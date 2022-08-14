from copyreg import pickle
import pandas as pd
import numpy as np

import os
import requests

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
# import spacy
import unidecode
from word2number import w2n
import contractions
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import math

import string

def convert_lower_case(data):
    return data.lower()


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

import unidecode
def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer

porter = PorterStemmer()


from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

# def remove_numbers(text):
#     no_number_string = re.sub(r'\d+','',text)
#     return no_number_string

def remove_numbers(text):
    no_number_string = re.sub(r'\d+','',text)
    return no_number_string


def preprocess(data):
    data = convert_lower_case(data)
    data = strip_html_tags(data)
    data = expand_contractions(data)
    data = remove_punctuation(data)
    # data = remove_numbers(data)
    # data = remove_accented_chars(data)
    data = remove_stop_words(data)
    data = stemming(data)
    return data

movie_reviews_df = pd.read_csv('movie_reviews_df.csv')


def doc_freq(word):
    
    DF = pd.read_pickle('DF.pkl')

    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


N = len(movie_reviews_df)



def D_array():

    N = len(movie_reviews_df)

    #with open('DF.pkl','rb') as handle:
        #DF = pickle.load(handle)

    DF = pd.read_pickle('DF.pkl')
    tf_idf = pd.read_pickle('TFIDF.pkl')

    #with open('TFIDF.pkl','rb') as handle:
        #tf_idf = pickle.load(handle)

    total_vocab = [x for x in DF]
    D = np.zeros((N, len(total_vocab)))

    for i in tf_idf:
        try:
            ind = total_vocab.index(i[1])
            D[i[0]][ind] = tf_idf[i]
        except:
            pass
    return D



def gen_vector(tokens):
    
    #with open('DF.pkl','rb') as handle:
        #DF = pickle.load(handle)
    
    DF = pd.read_pickle('DF.pkl')
    
    N = len(movie_reviews_df)

    total_vocab = [x for x in DF]
    Q = np.zeros((len(total_vocab)))

    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df1 = doc_freq(token)
        idf = math.log((N+1)/(df1+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim