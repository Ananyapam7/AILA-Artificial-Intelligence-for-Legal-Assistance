import string
import numpy as np
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(treebank_tag):
    """Return wordnet POS tagging for better wordnet lemmatization"""
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


def process(doc):
    """Takes a document and carries out lowercasing, stopword removal, lemmatization and Part Of Speech(POS) tagging to be utilized for doc2vec"""

    # For removal of stopwords
    stop_words = stopwords.words('english')

    # Adding extra stopwords that are masks
    extra_stop_words = ['[DATE]', '[TIME]', '[ADV]']
    stop_words.extend(extra_stop_words)
    stop_words = set(stop_words)

    # For lemmatization
    lemmatizer = WordNetLemmatizer()

    # Converting documents to lowercase
    text = word_tokenize(doc.lower())

    # Filtering out stopwords and punctuations
    text = list(filter(
        lambda token: token not in string.punctuation and token not in stop_words, text))

    # Getting POS tags for each word in the text
    pos_tags = pos_tag(text)

    # Putting proper wordnet POS tagging for better doc2vec performance
    text = [lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag))
            for (token, pos_tag) in pos_tags]

    # Returning a list containing the words of the passed document after processing
    return text
