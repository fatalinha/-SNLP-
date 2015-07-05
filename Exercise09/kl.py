#!/usr/bin/env python3
"""
Usage: ./kl.py

Exercise 9: Kullback-Leibler Divergence and Text Compression
Statistical Natural Language Processing

Anna Currey, 2554284
Alina Karakanta, 2556612
Kata Naszadi, 2556762
"""

from nltk.stem.snowball import EnglishStemmer
import math, os, string

# directory containing test texts
textdir = 'SupplementaryMaterial'

# punctuation to be removed from words
exclude = set(string.punctuation)

def processText(line):
    '''
    Lowercases, removes punctuation, and stems words in a line of text
    Parameter: line (string), the line of text to be processed
    Returns: words (list of strings), the processed words from the line
    '''
    words = []
    # break up the line and process each of the words
    for word in line.strip().split():
        # lowercase the words
        word = word.lower()
        # remove punctuation
        word = ''.join(char for char in word if char not in exclude)
        # stem words
        word = EnglishStemmer().stem(word)
        # now add to list of words
        words.append(word)
    return words

def kl_divergence(prob_dist):
	'''
	Computes KL divergence between two files
	'''
	pass
	

if __name__ == '__main__':
    # for each of the text files
    for filename in os.listdir(textdir):
        file = os.path.join(textdir, filename)
        
