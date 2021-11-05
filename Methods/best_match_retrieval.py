import argparse
import os
import string
from pathlib import Path
import pickle
import nltk
import numpy as np
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
import json
import helpers

def get_wordnet_pos(treebank_tag):
    """return wordnet POS tagging for better wordnet lemmatization"""
    if(treebank_tag.startswith('J')):
        return wordnet.ADJ
    elif(treebank_tag.startswith('V')):
        return wordnet.VERB
    elif(treebank_tag.startswith('N')):
        return wordnet.NOUN
    elif(treebank_tag.startswith('R')):
        return wordnet.ADV
    else:
        return wordnet.NOUN


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

path="./Resource/high_count_advs.json"
file_path="./Resource/masked_continuous_doc/"

print("training started....")
f = open (path, "r")
data = json.loads(f.read())

docs = []
Adv_name=[]
for key in data:
    for entry in data[key]['train']:
        _,case_no=entry.split('_')
        Adv_name.append(case_no)
        docs.append(open(file_path+case_no+".txt").read())


# Tokenizing the words and converting to lowercase
texts = [word_tokenize(doc.lower()) for doc in docs]

# Dropping all stopwords and punctuations
texts = [list(filter(lambda token: token not in string.punctuation and token not in stop_words, text)) for text in texts]

# Lemmatizing words with proper pos tagging
pos_tags = [nltk.pos_tag(text) for text in texts]

texts = [[lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag))
            for (token, pos_tag) in text_pos_tags] for text_pos_tags in pos_tags]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#bm_25obj = BM25Okapi(corpus)

save_model="./re/saved_model/"
filename = 'bm25_model.sav'
#pickle.dump(bm_25obj, open(save_model+filename, 'wb'))
bm_25obj=pickle.load(open(save_model+filename, 'rb'))

print("training complete....")
save="./re/deadline/"
print("testing started....")

result={}
for key in data:
    for entry in data[key]['test']:
        _,case_no=entry.split('_')
        test_content=open(file_path+case_no+".txt").read()
        texts = helpers.process(test_content)
        scores = bm_25obj.get_scores(texts)
        for score,adv in zip(scores,Adv_name):
            result[adv]={
                    case_no:score
        }
    
with open("check.json", "w") as write_file:
    json.dump(result, write_file,indent = 4)

            
            
print("testing complete....")
