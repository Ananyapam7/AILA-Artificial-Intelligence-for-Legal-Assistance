import string
import numpy as np
from scipy.spatial import distance

def softmax(array):
    """Returns the numerically stable softmax of a given array"""
    return (np.exp(array-np.max(array)))/np.sum(np.exp(array-np.max(array)))

def cosine_similarity(a, b):
    """Custom cosine similarity"""
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def norm_hamming(string1,string2):
    """Custom Normalized Hamming Distance"""
    return distance.hamming(list(string1), list(string2))

def jaccard_binary(x,y):
    """Returns the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity

def jaccard_similarity(doc1, doc2):
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split())

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)
