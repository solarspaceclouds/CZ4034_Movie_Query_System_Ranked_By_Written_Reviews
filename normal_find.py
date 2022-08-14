import string
from tokenize import String
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import re
import imdb
import array as arr
from sklearn.ensemble import RandomForestClassifier
from spellchecker import SpellChecker
import preprocess as pre
from nltk.tokenize import word_tokenize

st.write("""
# Movie Search Engine App
please query name, case sensitive as our app is still young and he wants the data as accurate as possible
""")

access = imdb.IMDb()
dfcolumns = pd.read_csv('all_data_with_sentiment.csv', nrows=1)
movie_reviews_df = pd.read_csv('all_data_with_sentiment.csv',
                                header=None,
                                skiprows=1,
                                usecols=list(range(len(dfcolumns.columns))),
                                names = dfcolumns.columns)

    

user_test = st.text_input(key='input', label ='input')
user_submit = st.button(label='Submit')


if user_submit:
    submit_df = movie_reviews_df.loc[movie_reviews_df['titles'].str.contains(user_test)]
    title_df = submit_df['titles'].unique()
    for i in title_df:
        url_df = movie_reviews_df.loc[movie_reviews_df['titles'].str.contains(i)]
        url_unique_df = url_df['url'].unique()
        id1 = re.findall('\d+', url_unique_df[0])
        id = int(id1[0])
        movie = access.get_movie(id)
        col1, col2, col3, col4 = st.columns(4)
        col1.image(movie['cover url'])
        col2.write(movie['title'])
        col3.write(url_unique_df[0])
        with col4:
            neg = submit_df.loc[(submit_df['titles'].str.contains(i)) & (submit_df['Predicted_Sentiment'] == 0) ]
            pos = submit_df.loc[(submit_df['titles'].str.contains(i)) & (submit_df['Predicted_Sentiment'] == 1) ]

            sizes = [len(neg), len(pos)]
            labels = 'Bad', 'Good'
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels = labels, autopct='%1.1f%%', startangle = 90)
            ax1.axis('equal')
            st.pyplot(fig1)
            
        

    #st.write(neg)
    #st.write(pos)