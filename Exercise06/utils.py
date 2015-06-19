# -*- coding: utf-8 -*-
"""
Helper functions for LM for IR exercise
Statistical Natural Language Processing

Anna Currey, 2554284
Alina Karakanta, 2556612
Kata Naszadi, 2556762
"""
from __future__ import division
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

text_begin = '.W'
doc_begin = '.I'
en_stopwords = stopwords.words("english")


# average precision for a query
def calc_avg_prec(returned_ranks, correct_ranks):
    '''
    returns: average precision (float) for the given query
    '''
    # number of relevant documents (normalize, and tell when to stop looking)
    all_docs = len(correct_ranks)
    # number of relevant found so far
    found_docs = 0
    curr_total = 0
    curr_index = 0  # rank looking at in the list
    
    # keep going until we get all the relevant ones
    while (found_docs != all_docs):
        # get the current doc
        curr_doc = returned_ranks[curr_index]
        # only care if relevant
        if curr_doc in correct_ranks:
            # we found one!
            found_docs += 1
            # find precision up until this rank
            curr_precision = found_docs / (curr_index + 1)
            # add it to the total
            curr_total += curr_precision
        # look at the next doc
        curr_index += 1
    
    # normalize total precision / number of docs
    return curr_total / all_docs


# precision calculation from last time
def calc_precision(query_ranks, gold_standard):
    '''
    returns: average precision measure given # of documents to return
    '''
    total_precision = 0
    num_queries = 0
    
    # check each of the queries
    for i in range(len(query_ranks)):
        num_queries += 1
        desired_retrieve = len(gold_standard[i])
        to_check = query_ranks[i][:desired_retrieve]
        retrieved = 0
        for doc in to_check:
            if doc in gold_standard[i]:
                retrieved += 1
        total_precision += retrieved / desired_retrieve
    
    # average
    return total_precision / num_queries


# how many times a word appears in the collection
def count_in_collection(word, word_docfreq):
    '''
    returns: number of times word appears in the collection
    '''
    word_count = 0
    if word in word_docfreq:
        for document in word_docfreq[word]:
            word_count += word_docfreq[word][document]
    return word_count


# note similar implementation as for exercise 5
# remove stopwords and lemmatizes (not given as options)
# added dictionary of document: word count
# word: doc: count instead of doc: word: count
def create_inverted_index(filename):
    '''
    returns
        dictionary inverted_index token: {doc_id: count}
        dictionary word_count docid: wordcount
    '''
    curr_doc = 0    # keep track of doc id
    inverted_index = {} # dict of word: doc: count in doc
    doc_wordcounts = {}     # dict of doc: wordcount
    
    in_text = False
    curr_wordcount = 0
    
    with open(filename, 'r') as file:
        for line in file:
            words = line.strip().split()
            # beginning of doc
            if words[0] == doc_begin:
                # update word count from last doc
                if curr_doc != 0:
                    doc_wordcounts[curr_doc] = curr_wordcount
                # in a new doc, not yet in text; also reset wordcount
                curr_doc += 1
                in_text = False
                curr_wordcount = 0
            
            # beginning of text
            elif words[0] == text_begin:
                # now in the text
                in_text = True
            
            # we're in the text an want to keep track of words
            elif in_text == True:
                # lemmatize, etc.
                words = normalize_text(words)
                # update wordcount
                curr_wordcount += len(words)
                # add words to inverted index
                for word in words:
                    if word in inverted_index:
                        if curr_doc in inverted_index[word]:
                            inverted_index[word][curr_doc] += 1
                        else:
                            inverted_index[word][curr_doc] = 1
                    else:
                        inverted_index[word] = {curr_doc: 1}
            
            # not in the text (superfluous text)
            else:
                pass
    # TO DO check for end of file (I cheated by adding .I at end..)
    
    return inverted_index, doc_wordcounts


# ranking of all documents in order of relevance to query
def doc_rank(query, word_docfreq, doc_size, lam, eps):
    '''
    returns: list containing all documents, ranked by relevance to query
    '''
    # normalize the query text
    query = normalize_text(query)
    
    # list of document probabilities for the query
    doc_probs = []
    
    # check each document
    for doc in doc_size:
        doc_probs.append((doc, 
                          prob_qd(query, doc, word_docfreq, doc_size, lam, eps)))

    # now get a ranking
    doc_ranking = sorted(doc_probs, key=lambda tup: tup[1], reverse=True)
    # only need the documents
    doc_ranking = [i[0] for i in doc_ranking]
    return doc_ranking


# number of tokens in collection
def get_collection_size(doc_size_dict):
    '''
    returns: number of words in the collection
    '''
    word_count = 0
    for document in doc_size_dict:
        word_count += doc_size_dict[document]
    return word_count


# vocabulary size
def get_vocab_size(word_docfreq_dict):
    return len(word_docfreq_dict.keys())

# mean average precision
def mean_avg_prec(query_ranks, gold_standard):
    '''
    returns: float for MAP of queries compared to gold standard
    '''
    # need sum of avg precision for each query
    avgp = 0
    num_queries = 0
    for query in query_ranks:
        num_queries += 1
        returned_ranks = query_ranks[query]
        correct_ranks = gold_standard[query]
        avgp += calc_avg_prec(returned_ranks, correct_ranks)
    # then normalize
    return avgp / num_queries
        

# removes punctuation and stopwords and performs stemming
# input: list of words to normalize
def normalize_text(list_of_words):
    '''
    returns: list of normalized words (punctuation and stopwords removed, stemmed)
    '''
    # punctuation removal, stopwords, stemming
    list_of_words = [w for w in list_of_words if w.isalpha()]
    list_of_words = [w for w in list_of_words if w not in en_stopwords]
    list_of_words = [EnglishStemmer().stem(w) for w in list_of_words]
    
    return list_of_words

# probability of query given document (product of probs of word given doc)
def prob_qd(query, document, word_docfreq, doc_size, lam, eps):
    '''
    returns: prob of query given doc
    '''
    curr_prob = 1
    for word in query:
        curr_prob *= prob_td(word, document, word_docfreq, doc_size, lam, eps)
    return curr_prob


# probability of term given document (ML with smoothing)
def prob_td(term, document, word_docfreq, doc_size, lam, eps):
    '''
    returns: prob of a term given the document
    '''
    # P_ML(t|d)
    try:
        ptd = word_docfreq[term][document] / doc_size[document]
    except:
        ptd = 0
    
    # P(t|C)
    coll_tf = count_in_collection(term, word_docfreq)
    coll_size = get_collection_size(doc_size)
    vocab_size = get_vocab_size(word_docfreq)
    ptc = (coll_tf + eps) / (coll_size + eps * vocab_size)
    
    # P(t|d) = (1-lambda)P(t|d) + lambdaP(t|C)
    return (1 - lam) * ptd + lam * ptc


