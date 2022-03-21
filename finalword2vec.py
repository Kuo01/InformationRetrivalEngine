# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:48:37 2021

@author: K
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
#nltk.download("punkt")
#nltk.download("stopwords")

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

stop_words = set(stopwords.words('english')) # get stop words
alph = set([chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)]) # add single characters into stop words
stop_words = stop_words.union(alph)
print(stop_words)


from os import listdir
filePath = 'E:/Master/Fall/InformationRetrievalSystems/Project/testdata/'
all_file=listdir(filePath) #获取文件夹中所有文件名
#for i in range(0,len(all_file)):
#print(i)
filename=all_file[0]   #获得文件名
filelabel=filename.split('.')[0]    #分离文件名和后缀


file_add =  filePath + filename     #完整的文件路径
f = open(file_add,'r',encoding='utf-8')
sentences=f.readlines()
    
f.close()  
#sentences = [str(df.loc[x, "full_text"]) for x in range(len(df))] # get all full text
# test = df.loc[0, "full_text"] # test data
sentences = [x.lower() for x in sentences] # lowercase
tk = RegexpTokenizer(r"[a-zA-Z]+") # remove numbers and punctuation
corpus = [tk.tokenize(x) for x in sentences] # tokenize
print(corpus)



for i in range(len(corpus)):
    corpus[i] = [x for x in corpus[i] if x not in stop_words] # remove stop words
print(corpus[0:1])

stemmer = PorterStemmer()
for i in range(len(corpus)):
    corpus[i] = [stemmer.stem(w) for w in corpus[i]]
print(corpus[0:1])

# model hyper parameters:
# Model type = skip-gram
# Sampling method = Negative Sampling
# window size = 10
# negative sample = 5
vec_size = 100
window_size = 10
negative_samples = 5
model = Word2Vec(sentences=corpus, 
                 vector_size = vec_size, 
                 window = window_size, 
                 min_count = 1, 
                 workers = 4, 
                 sg = 1, 
                 negative = negative_samples, 
                 hs = 1, 
                 epochs = 5)
# model.save("word2vec.model")
vocab = model.wv.index_to_key # get vocabulary
vec = {}
for w in vocab: # vector of related word
    vec[w] = model.wv.get_vector(w)
    
with open(r"E:/Master/Fall/InformationRetrievalSystems/project/w2v100.txt", 'w', encoding="utf-8") as f:
    f.write(str(len(vocab))+ " " + str(vec_size) + "\n") # first line: vocabulary size and vector size
    for w in vocab:
        line = w + " " + " ".join([str(x) for x in vec[w]]) + "\n"
        f.write(line)
    f.close()
    
query = "network"
print("Similiar words of " + query + " : ")
for i in model.wv.most_similar(query):
    print(i)
