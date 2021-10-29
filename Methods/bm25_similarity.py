# Script to compute the bm25 similarity scores between documents

import argparse
import os
import string
from pathlib import Path

import nltk
import numpy as np
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi



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


parser = argparse.ArgumentParser(
    description="Takes the root directory and finds the BM25 similarity scores between all cases files for each judge and stores them in a similar directory structure in the given output directory.")

parser.add_argument('-i', '--inputPath', required=True,
                    help="Root directory path containing subdirectories from which to use case documents for calculating BM25 scores", metavar="")
parser.add_argument('-o', '--outputPath',
                    help="Output root directory", metavar="")

args = vars(parser.parse_args())

# Getting a list of subdirectories under root directory and their full pathnames

subdir_paths = [f.path for f in os.scandir(args["inputPath"]) if f.is_dir()]


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


for subdir_path in subdir_paths:
    subdir_name = os.path.basename(subdir_path)
    docs = []
    filenames = []
    for path in Path(subdir_path).iterdir():
        filenames.append(os.path.basename(path))
        with open(path, "r") as f:
            docs.append(f.read())



    # Tokenizing the words and converting to lowercase
    texts = [word_tokenize(doc.lower()) for doc in docs]

    # Dropping all stopwords and punctuations
    texts = [list(filter(lambda token: token not in string.punctuation and token not in stop_words, text))
             for text in texts]

    # Lemmatizing words with proper pos tagging
    pos_tags = [nltk.pos_tag(text) for text in texts]

    texts = [[lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag))
              for (token, pos_tag) in text_pos_tags] for text_pos_tags in pos_tags]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm_25obj = BM25Okapi(corpus)

    for text, filename in zip(texts, filenames):
        tokenized_query = dictionary.doc2bow(text)

        # Getting the normalized BM25 scores
        scores = bm_25obj.get_scores(tokenized_query)
        scores = (scores - min(scores))/(max(scores) - min(scores))

        # Ranking the scores
        sorted_indices = np.argsort(-scores)
        ranked = np.empty_like(sorted_indices)
        ranked[sorted_indices] = np.arange(len(scores))

        # Writing to an output file
        output_file = os.path.join(
            args["outputPath"], "{}.txt".format(subdir_name))
        with open(output_file, "a") as f:
            f.write("Document: {}\n".format(filename))
            f.write("Doc\t\t\tRank\t\tBM25\n")
            for i, flname in enumerate(filenames):
                f.write("{}\t\t{}\t\t{}\n".format(
                    flname, ranked[i] + 1, scores[i]))
            f.write("\n\n")
