import string
#from pathlib import Path
#import pickle
import nltk
import os
#import numpy as np
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
import json
import helpers
from nltk.util import ngrams

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

def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data.lower()), num)
    return [ ' '.join(grams) for grams in n_grams]



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
#num=5
path="./high_count_advs.json"
file_path="./Resource/facts/"
save_model="./re/saved_model/"
#model_name = str(num)+"_gram"+'voting_new_bm25_model.sav'
save="./re/deadline/facts/"

f = open (path, "r")
data = json.loads(f.read())



docs = []
file_name=[]
skipfile=[]
less=[]
i=1

for key in data:
    file_name.append(key)
    file_content=""
    for entry in data[key]['train']:
        _,case_no=entry.split('_')                       
        try:         
            if os.stat(file_path+case_no+".txt").st_size>750:
                content=open(file_path+case_no+".txt").read()
                file_content+=content
            else:
                less.append(case_no) 
        except:
            skipfile.append(case_no)
            
    docs.append(file_content)
    #print("Total cases done {}".format(i))
    i+=1          

#print(less)
print(skipfile)

print("training started....")
texts = [word_tokenize(doc.lower()) for doc in docs]

# Dropping all stopwords and punctuations
texts = [list(filter(lambda token: token not in string.punctuation and token not in stop_words, text)) for text in texts]

# Lemmatizing words with proper pos tagging
pos_tags = [nltk.pos_tag(text) for text in texts]

texts = [[lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag))
            for (token, pos_tag) in text_pos_tags] for text_pos_tags in pos_tags]


#ngram_text=[]
#for text in texts:
#    x=' '.join(text)
#    ngram_text.append(extract_ngrams(x, num))




print("training complete....")
print("testing started....")
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
bm_25obj = BM25Okapi(corpus)
#pickle.dump(bm_25obj, open(save_model+model_name, 'wb'))
result={}
j=1
empty_file=[]
for key in data:
    for entry in data[key]['test']:
        _,case_no=entry.split('_')
        if os.stat(file_path+case_no+".txt").st_size <800:
            empty_file.append(case_no)
        else:
            test_content=open(file_path+case_no+".txt").read()
            query=helpers.process(test_content)
            #query=extract_ngrams(' '.join(query), num)
            tokenized_query = dictionary.doc2bow(query)
            scores = bm_25obj.get_scores(tokenized_query)
            scores = (scores - min(scores))/(max(scores) - min(scores))
            result[case_no]={}
            for score,file in zip(scores,file_name):
                result[case_no][file]=score    
            print("Total {} test cases done".format(j))
            j+=1
    
#print(empty_file)

with open('readme.txt', 'w') as f:
    for file in skipfile:
        f.write(file + '\n')

with open(save+"11new_fact_concatenated_scheme"+".json", "w") as write_file:
        json.dump(result, write_file,indent = 4)

