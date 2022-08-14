import string
from tokenize import String
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import imdb
import re
import array as arr
from sklearn.ensemble import RandomForestClassifier
from spellchecker import SpellChecker
import preprocess as pre
from nltk.tokenize import word_tokenize

st.write("""
# Movie Search Engine App
Please describe the movie and list the director/cast
""")


df = pd.DataFrame()
dfcolumns = pd.read_csv('written_ratings.csv', nrows=1)
movie_reviews_df = pd.read_csv('written_ratings.csv',
                                header=None,
                                skiprows=1,
                                usecols=list(range(len(dfcolumns.columns))),
                                names = dfcolumns.columns)
new_movie_reviews_df = pd.read_csv('all_data_with_sentiment.csv')

def cosine_similarity(k, query):

    preprocessed_query = pre.preprocess(query)

    tokens = word_tokenize(str(preprocessed_query))
    
    d_cosines = []
    
    query_vector = pre.gen_vector(tokens)
    
    D = pd.read_pickle('D.pkl')

    for d in D:
        d_cosines.append(pre.cosine_sim(query_vector, d))

    # 'out': is the list of indexes of top 10 movies with highest cosine scores (ranked) i.e. most similar to query
    out = np.array(d_cosines).argsort()[::-1]
    d_cosines.sort(reverse=True)
    df['cosine_scores'] = d_cosines
    df['movie_index_list'] = out 
    movie_title_list = []
    for i in out:
        movie_title_list.append(movie_reviews_df['titles'][i])
    df['movie_title'] = movie_title_list
    df1 = df.head(k)
    return df1


access = imdb.IMDb()

user_test2 = st.text_input(key="input2", label="")
query2 = pre.correct_spellings(user_test2)
top_movies_list = cosine_similarity(10,query2)
user_submit2 = st.button(label='Submit',key='submit2')
if user_submit2:
    for i in top_movies_list['movie_title']:
        name = i
        print (name)
        new_df = new_movie_reviews_df.loc[new_movie_reviews_df['titles'].str.contains(name)]
        title_df = new_df['titles'].unique()
        url_df = new_movie_reviews_df.loc[new_movie_reviews_df['titles'].str.contains(name)]
        url_unique_df = url_df['url'].unique()
        id1 = re.findall('\d+', url_unique_df[0])
        id = int(id1[0])
        movie = access.get_movie(id)
        col1, col2= st.columns([1,3])
        with col1:
            st.image(movie['cover url'])
        with col2:
            count = 0
            st.write(movie['title'])
            st.write(url_unique_df[0])
            #neg = new_df.loc[(new_df['titles'].str.contains(name)) & (new_df['Predicted_Sentiment'] == 0) ]
            pos = new_df.loc[(new_df['url'].str.contains(url_unique_df[0])) & (new_df['Predicted_Sentiment'] == 1) ]          
            count = len(pos) * 4
            st.write( str(count), ' percents of people likes this movie.')





    
