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
    for word, id in wordIds.items():
        print(word, id)

    return invertedIndex, wordIds



# for term count in mv index.items():
#   tuple(zip(*dterm.items()))

if __name__=="__main__":
    createInvertedIndex("cran/cran.all.1400")