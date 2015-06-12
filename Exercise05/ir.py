#!/usr/bin/env python3
"""
Usage:

Information retrieval exercise
Statistical Natural Language Processing

Anna Currey, 2554284
Alina Karakanta, 2556612
Kata Naszadi, 2556762
"""
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from collections import Counter
import math
import numpy as np

stopwords=stopwords.words("english")

def createInvertedIndex(filename, removeStop=False, lemmatize=False ):
    """
    returns
        dictionary wordId token:tokenId
        dictionary invertedIndex docid:token:count

    """
    currDoc=0
    invertedIndex={}
    file=open(filename, "r")
    read=True
    vocab=set()
    while read:
        line=file.readline()
        if line=="":
            break
        words=line.split()

        if words[0]==".W":
            currDoc+=1
            invertedIndex[currDoc]=[]
            words=file.readline().split()
            #.I stands for the end of the last document
            while words[0]!=".I" and read:
                #punctuation removal:
                words = [w for w in words if w.isalpha() ]
                if removeStop:
                    words=[w for w in words if w not in stopwords]
                if lemmatize:
                    words=[ EnglishStemmer().stem(w) for w in words]
                invertedIndex[currDoc]+=words
                #add new words to vocab
                vocab= vocab.union(set(words))
                line=file.readline()
                if line=="":
                    read=False
                else:
                    words=line.split()

    #convert list to dict word:count
    for id, list in invertedIndex.items():
        invertedIndex[id]=Counter(list)

    #make dict word:wordId
    wordIds=dict((v,i) for i,v in (enumerate(vocab)))

    return invertedIndex, wordIds

def createDocWordMatrix(invertedIndex,wordIds, tfIdf=True):

    def createIdf():
        """
        returns dict word:idf
        """
        idfs={}
        for word in wordIds:
            idfs[word]=0
            for docId in invertedIndex:
                doc=invertedIndex[docId]
                if word in doc:
                    idfs[word]+=1
            #if word is not OOV --> important for query idfs
            if idfs[word]:
                idfs[word]=math.log(len(invertedIndex)/idfs[word])

        return idfs

    docWordMatrix=np.zeros((len(invertedIndex), len(wordIds)), np.float64)
    idfs=createIdf()
    if tfIdf:
        for docId in invertedIndex:
            doc=invertedIndex[docId]
            #calculate the size of doc (sum of count of words)
            docSize=sum(doc.values())
            for word, wordId in wordIds.items():
                if word in doc:
                    docWordMatrix[docId-1, wordId]= (doc[word]/docSize) * idfs[word]
    #use termfrequencies
    else:
        for docId, doc in invertedIndex.items():
            for word, wordId in wordIds.items():
                if word in doc:
                    docWordMatrix[docId-1, wordId]= doc[word]



    return docWordMatrix

def similarity(docM, queryV):
    #find documentTfIdfs where query nonzero --> terms in query
    indexes=np.nonzero(queryV)[0]
    docs=docM[:, indexes]

    #take the sum for each document = each row
    sums=np.sum(docs,1)
    #its ascending, bec i dunno how to de descending :/
    orderedDocs=np.argsort(sums)+1

    return orderedDocs


def test_queries(tf_idf, remove_stop, lemmatize):
    docInvertedIndex, wordIds = createInvertedIndex("cran/cran.all.1400",remove_stop, lemmatize)
    docM=createDocWordMatrix(docInvertedIndex,wordIds, tf_idf)
    queryInvertedIndex, qwordIds=createInvertedIndex("cran/cran.qry", remove_stop, lemmatize)
    # we want the queryMatrix to be indexed the same way so we use wordIds and not qwordIds
    # OOVs will have 0 tfidfs, but they are not important for the similarity anyway
    queryM=createDocWordMatrix(queryInvertedIndex, wordIds, tf_idf)
    numQueries=queryM.shape[0]
    
    #get gold stanadard:
    target=np.loadtxt("cran/cranqrel", dtype=int)
    #disregard 3.column
    target=target[:,:2]

    # list of query results
    query_results = []
    for i in range(0,numQueries):
        #select docs relevant query i
        targetset=set(target[np.where(target[:,0]==i+1)][:,1])
        numOfTarget=len(targetset)
        queryVector=queryM[i,:]
        result=set(similarity(docM, queryVector)[-numOfTarget:])
        good=result.intersection(targetset)
        recall=len(good )/ numOfTarget
        #print(i+1, good, recall)
        query_results.append(recall)
    # now get the average
    average = np.mean(query_results)
    return average




if __name__=="__main__":
    # test each of the options
    # TF-IDF
    print('TF-IDF:')
    for remove_stop in [True, False]:
        for lemmatize in [True, False]:
            print('Stop words removed: ' + str(remove_stop) + 
                  ', Lemmatized: ' + str(lemmatize) + 
                  ', Avg precision: ' + str(test_queries(True, remove_stop, lemmatize)))
            #print(test_queries(True, remove_stop, lemmatize))
    # matrix
    print('Doc-word matrix:')
    for remove_stop in [True, False]:
        for lemmatize in [True, False]:
            print('Stop words removed: ' + str(remove_stop) + 
                  ', Lemmatized: ' + str(lemmatize) + 
                  ', Avg precision: ' + str(test_queries(False, remove_stop, lemmatize)))




