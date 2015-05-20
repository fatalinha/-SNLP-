#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO what this does
TODO usage

TODO input and output file formats

22 May 2015
Anna Currey, Alina Karakanta, Kata Naszadi
"""
import matplotlib.pyplot as pyplot
import os
from math import log

## constants needed
# directories containing spam and ham training
# change as appropriate so it works on your system
spam_train_dir = '../Ex2SuppMaterial/dataset/spam-train/'
ham_train_dir = '../Ex2SuppMaterial/dataset/nonspam-train/'

# number of training files
num_spam_train = len(os.listdir(spam_train_dir))
num_ham_train = len(os.listdir(ham_train_dir))
num_total_train = num_spam_train + num_ham_train

# words to experiment with
words = ['free', 'money', 'cash', 'save', 'want', 'language', 'university', 
         'linguistic', 'click', 'internet']

## functions
# returns the information gain of a given class and word
# TODO need to check calculations
def computeInfGain(word):
    # compute -sum(P(ci)logP(ci))
    spam_prob = prob_class(spam_train_dir)
    ham_prob = prob_class(ham_train_dir)
    part1 = -1 * (spam_prob * log(spam_prob) + ham_prob * log(ham_prob))
    # compute P(t)sum(P(ci|t)logP(ci|t))
    spamword_prob = prob_classterm(spam_train_dir, ham_train_dir, word)
    hamword_prob = prob_classterm(ham_train_dir, spam_train_dir, word)
    word_prob = prob_term(word)
    part2 = word_prob * (spamword_prob * log(spamword_prob) + \
            hamword_prob * log(hamword_prob))
    # compute P(not t)sum(P(ci|not t)logP(ci|not t))
    part3 = (1 - word_prob) * ((1 - spamword_prob) * log(1 - spamword_prob) + \
            (1 - hamword_prob) * log(1 - hamword_prob))
    # IG = part1 + part2 + part3
    return part1 + part2 + part3
 
# returns the mutual information of a given class and word
def computeMutInf(my_class_dir, other_class_dir, word):
    # compute log P(c | t)
    classterm_prob = log(prob_classterm(my_class_dir, other_class_dir, word))
    # compute log P(c)
    class_prob = log(prob_class(my_class_dir))
    # I(t, c) = log(P(t|c)) - log(P(t))
    return classterm_prob - class_prob

# returns the document frequency of a term in a given class
def freq_termclass(my_class_dir, my_term):
    freq = 0
    for curr_file in os.listdir(my_class_dir):
        # check if the term is in that file
        # TO DO might have to change this is files are too large (memory)
        if my_term in open(my_class_dir + curr_file).read():
            freq += 1
    return freq

# plots the words with their mutual info / info gain
def make_plot(data_list, figure_name):
    figure = pyplot.figure()
    pyplot.bar(range(len(words)), data_list)
    pyplot.xticks(range(len(words)), words, rotation=25)
    pyplot.savefig(figure_name)
    return

# returns P(c = spam)
def prob_class(my_class_dir):
    if my_class_dir == spam_train_dir:
        return num_spam_train / num_total_train
    elif my_class_dir == ham_train_dir:
        return num_ham_train / num_total_train
    else:
        print('Class must be either ham or spam.')
        return 1

# returns P(class | term)
def prob_classterm(my_class_dir, other_class_dir, my_term):
    # count(class, term)
    classterm_count = freq_termclass(my_class_dir, my_term)
    # count(term)
    term_count = classterm_count + freq_termclass(other_class_dir, my_term)
    # P(class | term) = count(class, term) / count(term)
    return classterm_count / term_count

# returns P(term)
def prob_term(my_term):
    freq = 0
    # add spam doc freq
    freq += freq_termclass(spam_train_dir, my_term)
    # add ham doc freq
    freq += freq_termclass(ham_train_dir, my_term)
    # P(term) = total doc freq / number of training docs
    return freq / num_total_train


## main program
if __name__ == '__main__':
    # will test on each of words and store results in lists
    # TODO should this actually be tested on the test data?
    info_gains = []
    mut_infos = []
    for word in words:
        info_gains.append(computeInfGain(word))
        # note just do mutual info for spam
        mut_infos.append(computeMutInf(spam_train_dir, ham_train_dir, word))
    # make graphs
    # TODO axis labels
    make_plot(info_gains, 'information-gain.pdf')
    make_plot(mut_infos, 'mutual-information.pdf')
    