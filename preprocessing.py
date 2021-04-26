import glob
import functools
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import re
import numpy as np
import pandas as pd
import os

# Data Exploration

import csv
read_files = glob.glob('/kaggle/input/legalai/Object_casedocs/*')
with open("object_casedocs.csv", "w") as outfile:
    w=csv.writer(outfile)
    for f in read_files:
        with open(f, "r") as infile:
            w.writerow([" ".join([line.strip() for line in infile])])


lst_arr = os.listdir('/kaggle/input/legalai/Object_casedocs/')
df_filename = pd.DataFrame(lst_arr, columns = ['Name'])

evaluate = pd.read_csv('/kaggle/input/legalai/relevance_judgments_priorcases.txt', delimiter = " ", header = None)
evaluate.columns = ["Query_Number", "Q0", "Document" ,"Relevance"]
evaluate=evaluate.drop(columns=["Q0"])

df = pd.read_csv('object_casedocs.csv',header=None)
df.columns = ["Text"]


df = pd.concat([df_filename,df], axis = 1)

print(len(df))
print(df.shape)
print(df.info)

#Preprocessing

import re
#Convert lowercase remove punctuation and Character and then strip
text = df.iloc[0]
print(text)
text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
txt = text.split()
print(txt)

#remove stopwords
import nltk
lst_stopwords = nltk.corpus.stopwords.words("english")
txt = [word for word in txt if word not in lst_stopwords]
print(txt)

#stemming
ps = nltk.stem.porter.PorterStemmer()
print([ps.stem(word) for word in txt])

#Lemmetization
nltk.download('wordnet')
lem = nltk.stem.wordnet.WordNetLemmatizer()
print([lem.lemmatize(word) for word in txt])

def utils_preprocess_text(text, flg_stemm=True, flg_lemm =True,lst_stopwords=None):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    #tokenization(convert from string to List)
    lst_text = text.split()
    #remove stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    #stemming
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    #Lemmentization
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)

    return text


df['clean_text'] = df['Text'].apply(lambda x: utils_preprocess_text(x,flg_stemm = False, flg_lemm=True))

train = df["clean_text"]

test = pd.read_csv("/kaggle/input/legalai/Query_doc.txt",delimiter ="|",header=None)
test.columns = ["AILA","NAN", "Query"]
test=test.drop(columns=["AILA","NAN"])

test['Query_processed'] = test['Query'].apply(lambda x:utils_preprocess_text(x, flg_stemm = False, flg_lemm=True))
