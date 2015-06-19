#!/usr/bin/env python3
"""
Usage: ./lm_ir.py

LM for IR exercise
Statistical Natural Language Processing

Anna Currey, 2554284
Alina Karakanta, 2556612
Kata Naszadi, 2556762
"""
import utils

doc_filename = 'cran/cran.all.1400'
query_filename = 'cran/cran.qry'
gold_filename = 'cran/cranqrel'

lam = 0.5
eps = 0.0001

query_start = '.I'
text_start = '.W'

# TO DO -- fix the whole .I thing


def main():
    # get the dictionaries: word: {docId: count} and doc: size
    word_docfreq, doc_size = utils.create_inverted_index(doc_filename)
    
    # read in the queries and get the rankings
    # dict query_id: list of documents in order
    query_ranks = {}
    curr_query_id = 0
    curr_query_words = []
    #in_query = False
    
    with open(query_filename, 'r') as query_file:
        for line in query_file:
            words = line.strip().split()
            # find the start of a query
            # TO DO cheated by putting .I at end of doc
            if words[0] == query_start:
                # we have found all words in prev query so can check it
                if curr_query_id > 0:
                    doc_ranks = utils.doc_rank(curr_query_words, word_docfreq, 
                                               doc_size, lam, eps)
                    query_ranks[curr_query_id] = doc_ranks
                # in a new doc, not yet in query
                curr_query_id += 1
                #in_query = False
                curr_query_words = []
                
            # ignore lines that begin with .W
            elif words[0] != text_start:
                # add the words to the query
                for word in words:
                    curr_query_words.append(word)
    
    # read in gold standard for queries
    # keep track of query: doc ranks
    gold_standard = {}
    with open(gold_filename, 'r') as gold_file:
        for line in gold_file:
            # lines are queryID: docID
            # can use query ID here because it starts from 1 as desired
            parts = line.strip().split()
            if int(parts[0]) in gold_standard:
                gold_standard[int(parts[0])].append(int(parts[1]))
            else:
                gold_standard[int(parts[0])] = [int(parts[1])]
    
    # calculate and print MAP
    mean_avg_prec = utils.mean_avg_prec(query_ranks, gold_standard)
    print('Mean avg precision: ' + str(mean_avg_prec))
    
    # calculate and print precision from last time
    precision = utils.calc_precision(query_ranks, gold_standard)
    print('Average precision: ' + str(precision))


if __name__ == '__main__':
    main()