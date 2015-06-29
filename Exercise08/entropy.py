#!/usr/bin/env python3
"""
Usage: TODO

Exercise 8: Entropy
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

def Entropy(prob_dist):
    '''
    Outputs the entropy for the probability distribution
    Parameter: prob_dist (dictionary), dict of token to probability
    Returns: entropy (float), entrpy of the probability distribution
    '''
    entropy = 0
    # TODO
    # for each word in the distribution
        # get p(word)
        # get log(p(word)) (use base 2)
        # add -p(w)log(p(w)) to the entropy running total
    return entropy


def parseText(filename):
    '''
    Outputs a probability distribution for a text file
    Parameter: filename (string), the name of the file to be parsed
    Returns: prob_dist (dictionary), {token: probability} dictionary
    '''
    # keeps track of token: count
    count_dict = {}
    # total number of words in the file
    num_words = 0
    # TODO
    # open the file
        # loop through, line by line
            # process the line and get list of words
            # for each of the words
                # increment the total number of words in the file
                # if word is in the dictionary, update its count
                # otherwise, add it to the dictionary with a count of 1
    # now convert to a count dict to probability distribution
    prob_dist = {} # TODO
    return prob_dist


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
    return words


if '__name__' == '__main__':
    # for each of the text files
    for filename in os.listdir(textdir):
        file = os.path.join(textdir, filename)
        # parse it
        prob_dist = parseText(file)
        # get the entropy
        entropy = Entropy(prob_dist)
        # print it out
        print('Entropy for {}: {}'.format(filename, entropy))