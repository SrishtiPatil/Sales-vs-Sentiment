# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:40:13 2020

@author: Srishti
"""

import numpy as np
import pandas as pd
import re
import nltk

file = './Data Final.xlsx'
xls = pd.ExcelFile(file)
df = xls.parse('2015')


#Null Values removal
df.isnull().sum()
df_no_nan=df.dropna()
df_no_nan.isnull().sum()


#Lower case
df_temp = df_no_nan['message'].str.lower()
df_no_nan['message_lower']=df_temp
df_no_nan['message_lower'].head()


#Punctuation removal
df_temp = df_no_nan['message_lower'].str.replace('[^\w\s]','')
df_no_nan['text_punct']=df_temp
df_no_nan['text_punct'].head



#stop word removal

#Importing stopwords from nltk library
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
# Function to remove the stopwords
def stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
# Applying the stopwords to 'text_punct' and store into 'text_stop'
df_temp = df_no_nan['text_punct'].apply(stopwords)
df_no_nan['text_stop']=df_temp
df_no_nan["text_stop"].head()


#Emoticon removal
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
# Function for removing emoticons
def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

# applying remove_emoticons to 'text_rare'
df_no_nan['text_rare'] = df_no_nan['text_stop'].apply(remove_emoticons)

#Common words
from collections import Counter
cnt = Counter()
for text in df_no_nan['text_stop'].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(20)

nltk.download('punkt')
# importing word_tokenize from nltk
from nltk.tokenize import word_tokenize
# Passing the string text into word tokenize for breaking the sentences
token = word_tokenize(text)
token
df_no_nan.to_excel("2015 no emot.xlsx") 


file = './2015 no emot.xlsx'
xls = pd.ExcelFile(file)
df = xls.parse('Sheet1')
#Creating function for tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text
# Passing the function to 'text_rare' and store into'text_token'
df['text_token'] = df['text_rare'].apply(lambda x: tokenization(str(x).lower()))
df[['text_token']].head()
df.to_excel("2015 no emot.xlsx") 

from textblob import TextBlob
df['Polarity']=df['text_rare'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df.head()

df.to_excel("2015 Polarity.xlsx") 
file = './2015 Polarity.xlsx'
xls = pd.ExcelFile(file)
df = xls.parse('Sheet1')
df['Polarity_Category']=''
df['Polarity_Category']=np.where(((df['Polarity']<=-0.49) & (df['Polarity']>=-1)), 'Very Negative',df['Polarity_Category'])
df['Polarity_Category']=np.where(((df['Polarity']<0) & (df['Polarity']>=-0.5)), 'Negative',df['Polarity_Category'])
df['Polarity_Category']=np.where(df['Polarity']==0, 'Neutral',df['Polarity_Category'])
df['Polarity_Category']=np.where(((df['Polarity']<=0.49) & (df['Polarity']>=0.01)), 'Positive',df['Polarity_Category'])
df['Polarity_Category']=np.where(((df['Polarity']<=1) & (df['Polarity']>=0.49)), 'Very Positive',df['Polarity_Category'])

df.head()
df.to_excel("2015 Polarity.xlsx") 



'''from translate import Translator
translator= Translator(to_lang='English')
a=[]
for i in df_no_nan['text_stop']:
    a.append(translator.translate(i))
a.to_excel("trans.xlsx") '''

