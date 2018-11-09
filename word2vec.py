import pandas as pd 
import numpy as np 
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

#df = pd.read_csv("training.1600000.processed.noemoticon.csv",encoding = " ISO-8859-1")
#X = df.iloc[:,5]
#
#
##Remove the links and tags 
#clean = []
#for sentence in X:      
#    temp = []
#    sentence = sentence.lower()
#    sentence = sentence.split()
#    for word in sentence:
#        if  word[0]== '@':
#            continue
#        if  word[0]== '#':
#            continue
#        if len(word)>9:
#            if word[0]=='h' and word[1] =='t' and  word[2] =='t' and word[3]=='p' :
#                continue
#            if word[0] == 'w' and word[1] == 'w' and word[2] == 'w' and word[3] == '.':
#                continue
#        temp.append(word)
#    temp = ' '.join(temp)
#    clean.append(temp)
#clean = pd.DataFrame(clean)
#clean.to_csv("clean.csv",header = 'Review')


df = pd.read_csv("clean.csv",encoding = " ISO-8859-1")
X = df.iloc[:,1]
del(df)
del(corpus)
corpus = []
for sentence in X:
    if type(sentence) == type('str'):
        review = re.sub('[^a-zA-Z]', ' ',sentence)
        corpus.append(review)


import gensim 
from nltk.tokenize import word_tokenize
import multiprocessing 
tok=[word_tokenize(sent) for sent in corpus]

vocab_size = 600
num_workers = multiprocessing.cpu_count()
min_counts = 5
down_sample = 1e-2
context =   5

model = gensim.models.Word2Vec(tok,
                     size = vocab_size,
                     workers =num_workers,
                     window = context,
                     min_count = min_counts,
                     sample = down_sample)

model.most_similar('awesome')

model.init_sims(replace=True)
model_name = "English Twitter"
model.save(model_name)

