#! /usr/bin/env python3
'''
Created on Jun 17, 2015

@author: janis
'''
from math import log,exp

def invert_dict(d):
    '''
        Helper function
        Converts a translation count dict from cEGFNZ to cFGENZ and vice versa
        (here c denotes count instead of probability)
        Also handles the NULL entries, to which it assigns a count of 1. 
    '''
    d = d.copy()
    del d['NULL']
    invertedDict = {'NULL': dict( (key,1) for key in d) }
    for keyS,values in d.items():
        for keyT, count in values.items():
            if keyT not in invertedDict:
                invertedDict[keyT] = {}
            invertedDict[keyT].update({keyS:count})
    return invertedDict

def add_null_token(cNZ):
    '''
        Helper function (ignore)
        Add a NULL key with unit count to each other target token.
    '''
    from itertools import chain
    targetTokens = chain(*map(dict.keys, cNZ.values()))
    cNZ['NULL'] = dict( (targetToken,1) for targetToken in targetTokens)

def normalize_counts(conditionalFrequencies, conditionCounts = None):
    '''
        Helper function
        Normalized counts of dict of dicts (think: cFGENZ) to create probabilities.
        
        If conditionCounts is not provided, the resulting dictionaries are normalized
        to add up to one (creating fully correct conditional probabilities).
        If a unigram count is provided instead, the counts are normalized w.r.t. these counts.
        In our case the latter gives more realistic counts (based on the Europar corpus),
        but you may be curious to experiment as you wish. 
    '''
    if conditionCounts is None:
        conditionCounts = dict( (condition, sum(counts.values())) for condition,counts
                                in conditionalFrequencies.items())
    conditionalProbabilities = dict()
    for condition, values in conditionalFrequencies.items():
        # Ignore OOV
        if condition not in conditionCounts:
            continue
        conditionCount = conditionCounts[condition]
        conditionalProbabilities[condition] = dict( (x,count/conditionCount) for x,count in values.items())
    return conditionalProbabilities
    
def print_ranking(ranking, k = 4):
    '''
        Helper function (ignore)
        Prints the topmost k entries of a ranking
    '''
    if k is None or k<0:
        k = len(ranking)
    lines = []
    for i in range(0, min(k,len(ranking))):
        element = ranking[i][0]
        elementString = ' '.join(map(lambda elem:'{0:2}'.format(elem), element))
        score = ranking[i][1]
        line = '{0} {1:10.4g}: ({2})'.format(i+1,score, elementString)
        lines.append(line)
    message = '\n'.join(lines)
    print(message)
    return message
    
def get_all_alignments(l,m):
    '''
        Helper Function (ignore)
        Retrieves a generator of all possible alignments of a source sentence of size l and a
        destination sentence of size m.
        Note that this generator will give potentially too many entries
        e.g. len(tuple(get_alignments(5,4)) == 1296
        
        The elements of the generator are m-sized tuples, which at each element have an
        index between -1,0,...,m-1. The index -1 denotes a NULL token.   
    '''
    for index in range(0,(l+1)**m):
        a = []
        remainder = index
        for _ in range(0,m):
            a.append(remainder%(l+1)-1)
            remainder = int(remainder/(l+1))
        yield a
    return None

def iterate_over_sets(allowedSets):
    '''
        Helper function. (ignore)
        Given a tuple of tuples, this function returns a generator that will produce
        all combinations of tuples t of size equal to len(allowedSets).
        Each entry t_i in the returned tuple t will take on values from the iterable in
        allowedSets[i].
        
        e.g.: tuple(iterate_over_sets([['1A','1B'],['2C','2D',3F']])) will give the 6 tuples
        (('1A', '2C'), ('1B', '2C'), ('1A', '2D'), ('1B', '2D'), ('1A', '3F'), ('1B', '3F'))
    '''
    m = len(allowedSets)
    numberOfAllowedIndices = tuple(map(len,allowedSets))
    indices = [0]*m
    # While the most significant index is not overflown
    while True:
        for i in range(0,m):
            if indices[i] == numberOfAllowedIndices[i]:
                # Overflow
                indices[i] = 0
                # The last index overflowed. We are done.
                if i == m-1:
                    return None 
                indices[i+1] += 1
            else:
                break
        # convert index within allowed indices to set elements.
        output = tuple(allowedSets[i][indices[i]] for i in range(0,m))
        indices[0] += 1
        yield output
    return None

def get_valid_alignments(sentenceE, sentenceF, tFGENZ):
    '''
        Returns a generator that will produce only the valid alignments for the two sentences
        provided.
        
        Parameters
        ----------
        sentenceE: an iterable of words
            The source sentence
        
        sentenceF: an iterable of words
            The destination sentence
        
        tFGENZ: a word-keyed dict of word-keyed dicts of probabilities
            The translation probabilities of a foreign token given the source (English) token,
            that only contains entries for the non-zero (NZ) translations.
        
        Returns
        -------
        generator of tuples:
            each returned tuple is a tuple of size len(sentenceF) that represents an alignment.
            All alignments a between the two sentences are returned, as long as there is a supporting 
            non-zero probability of transmission. For each alignment it holds that:
            all(tFGENZ[sentenceEN[a[i]]][sentenceF[i]] for i in range(0,len(sentenceF)))
            where sentenceEN = list(sentenceE) + ['NULL'] 
    '''
    # Create token to index mapping
    tokenIndicesE = dict()
    for idxE,wordE in enumerate(sentenceE):
        tokenIndicesE[wordE] = tokenIndicesE.get(wordE,[]) + [idxE]
    tokenIndicesF = dict()
    for idxF,wordF in enumerate(sentenceF):
        tokenIndicesF[wordF] = tokenIndicesF.get(wordE,[]) + [idxF]
    allowedTokenIndices = dict( (existingTokenF,[-1]) for
                                existingTokenF in tokenIndicesF.keys() )
    tokensF = set(tokenIndicesF.keys())
    for tokenE in tokenIndicesE.keys():
        validTargetsF = tFGENZ[tokenE]
        existingTargetTokensF = tokensF.intersection(set(validTargetsF))
        for tokenF in existingTargetTokensF:
            existingSourceIndices = list(tokenIndicesE[tokenE])
            allowedTokenIndices[tokenF] += existingSourceIndices
    allowedWordIndices = tuple( allowedTokenIndices[wordF] for wordF in sentenceF )
    return iterate_over_sets(allowedWordIndices)

def get_valid_translations_of_alignment(sentenceE, alignment, tFGENZ):
    '''
        Returns a generator that will produce only the valid translations for a specific alignment
        and a source (English) sentenceE. 
        
        Parameters
        ----------
        sentenceE: an iterable of words
            The source sentence
        
        alignment: an iterable of indices
            The alignment between the requested translations and the source sentence
        
        tFGENZ: a word-keyed dict of word-keyed dicts of probabilities
            The translation probabilities of a foreign token given the source (English) token,
            that only contains entries for the non-zero (NZ) translations.
        
        Returns
        -------
        generator of tuples:
            each returned tuple is a tuple of size len(alignment) that represents a possible translated
            sentence in the foreign language, sentenceF.
            All possible translations of sentenceE are returned, as long as there is a supporting 
            non-zero probability of transmission for the given alignment. For each translation
            sentenceF holds that:
            all(tFGENZ[sentenceEN[a[i]]][sentenceF[i]] for i in range(0,len(alignment))) 
            where sentenceEN = list(sentenceE) + ['NULL'] 
    '''
    allowedTokensF = []
    for indexE in alignment:
        wordE = sentenceE[indexE]
        tokensF = tuple(tFGENZ[wordE].keys())
        allowedTokensF.append(tokensF)
    return iterate_over_sets(allowedTokensF)

def get_valid_translations(sentenceE, m, tFGENZ):
    '''
        Returns a generator that will produce only the valid translations for a specific
        size m of a destination sentence for a given source (English) sentenceE. 
        
        Parameters
        ----------
        sentenceE: an iterable of words
            The source sentence
        
        m: positive integer
            The length of the returned translations len(sentenceF)
        
        tFGENZ: a word-keyed dict of word-keyed dicts of probabilities
            The translation probabilities of a foreign token given the source (English) token,
            that only contains entries for the non-zero (NZ) translations.
        
        Returns
        -------
        generator of tuples:
            each returned tuple is a tuple of size m that represents a possible translated
            sentence in the foreign language, sentenceF, under some (not given) alignment a.
            All possible translations of sentenceE are returned, as long as there is a supporting 
            non-zero probability of transmission for the given alignment. For each translation
            sentenceF holds that:
            all(tFGENZ[sentenceEN[a[i]]][sentenceF[i]] for i in range(0,len(alignment))) 
            where sentenceEN = list(sentenceE) + ['NULL'] 
    '''
    tokensE = set(sentenceE)
    allTokensF = []
    for wordE in tokensE:
        tokensF = list(tFGENZ[wordE].keys())
        allTokensF += tokensF
    allowedTokensF = [tuple(set(allTokensF))] * m
    return iterate_over_sets(allowedTokensF)

def sentence_score_f_a_given_e(sentenceE, sentenceF, alignment, tokenProbabilityFGivenE):
    '''
        Computes the *unnormalized* probability of a target sentence and a specific given alignment
        that could explain a given source (English) sentence \hat P(f,a|e). In latex is is
            \hat P(f,a|e) = 1/(l+1)^m *\prod_{i=1}^m( t(f_i|e_{a_i})
            and the probability P(f,a|e) = \epsilon \hat P(f,a|e) for some constant \epsilon. 
        
        Parameters
        ----------
        sentenceE: an iterable of words
            The source sentence
        
        sentenceF: an iterable of words
            The destination sentence

        alignment: an iterable of indices
            The alignment between the requested translations and the source sentence
        
        tFGENZ: a word-keyed dict of word-keyed dicts of probabilities
            The translation probabilities of a foreign token given the source (English) token,
            that only contains entries for the non-zero (NZ) translations.
        
        Returns
        -------
        float: 
            The unnormalized probability of f and a given e \hat P(f,a|e).
    '''
    l = len(sentenceE)
    m = len(sentenceF)
    sentenceEN = list(sentenceE) + ['NULL']
    '''######### TODO: #########
    Complete this function.
    Unless you want to be very explicit, this can be adequately completed in a few (4-5) lines.
    Note that you do not need to deal with OOV words, unless you change the dataset.
    '''
    prob = 0 # also consider log-scale probabilities.
    return prob
    
def sentence_score_f_given_e(sentenceE, sentenceF, tFGENZ):
    '''
        Computes the *unnormalized* probability of a target sentence that could explain a given
        source (English), over *all* possible alignments, \hat P(f|e). In latex is is
            \hat P(f|e) = \sum_a \hat P(f,a|e)
            and the probability P(f|e) = \epsilon \hat P(f|e) for some constant \epsilon. 
        
        Parameters
        ----------
        sentenceE: an iterable of words
            The source sentence
        
        sentenceF: an iterable of words
            The destination sentence

        tFGENZ: a word-keyed dict of word-keyed dicts of probabilities
            The translation probabilities of a foreign token given the source (English) token,
            that only contains entries for the non-zero (NZ) translations.
        
        Returns
        -------
        float: 
            The unnormalized probability of f given e \hat P(f|e).
    '''
    alignmentsValid = get_valid_alignments(sentenceE, sentenceF, tFGENZ)
    prob = 0
    '''######### TODO: #########
     Complete this function.
    If you need more than a couple of lines, get really suspicious.
    '''
    return prob

def sentence_score_f_given_e_fast(sentenceE, sentenceF, tokenProbabilityFGivenE):
    '''
        Computes the *unnormalized* probability of a target sentence that could explain a given
        source (English), over *all* possible alignments, \hat P(f|e). In latex is is
            \hat P(f|e) = \sum_a \hat P(f,a|e)
            and the probability P(f|e) = \epsilon \hat P(f|e) for some constant \epsilon. 
        
        This is a fast implementation that exploits an algebraic property of the resulting 
        product-of-sums to substantially drop the complexity of the computation.
        
        Parameters
        ----------
        sentenceE: an iterable of words
            The source sentence
        
        sentenceF: an iterable of words
            The destination sentence

        tFGENZ: a word-keyed dict of word-keyed dicts of probabilities
            The translation probabilities of a foreign token given the source (English) token,
            that only contains entries for the non-zero (NZ) translations.
        
        Returns
        -------
        float: 
            The unnormalized probability of f given e \hat P(f|e).
    '''
    l = len(sentenceE)
    m = len(sentenceF)
    sentenceE = list(sentenceE) + ['NULL']
    logprob = -m*log(l+1)
    for j in range(0,m):
        wordF = sentenceF[j]
        terms = tuple( (tokenProbabilityFGivenE[wordE].get(wordF,0)) for wordE in sentenceE)
        sumTerms = sum(terms)
        if not sumTerms == 0:
            logprob += log(sumTerms)
    return exp(logprob)


def rank_valid_alignments(sentenceE, sentenceF, tEGFNZ):
    '''
        Iterate over all alignments, evaluate some score on them and rank them in decreasing probability.
        
        Parameters
        ----------
        sentenceE: an iterable of words
            The source sentence
        
        sentenceF: an iterable of words
            The destination sentence

        tFGENZ: a word-keyed dict of word-keyed dicts of probabilities
            The translation probabilities of a foreign token given the source (English) token,
            that only contains entries for the non-zero (NZ) translations.
        
        Returns
        -------
        iterable of
            2-tuple: pair of alignment and score
            Return all (valid) alignments scored and ranked.
            e.g.: ( ((0,1),0.5), (-1,0), 0.4 ), ... )
                     ^   ^  ^    ^    ^   ^ 
                     +---+--|----+----+---|--------------- Alignment
              score ------- +-------------+      
    '''
    alignments = get_valid_alignments(sentenceE, sentenceF, tEGFNZ)
    '''######## TODO #########
    Complete this function.
    You can should be able to do this in no more that 2-3 lines.
    '''
    scoreAlignments = []
    sortedScoreAlignments = sorted(scoreAlignments, key=lambda pair:-pair[1])
    return sortedScoreAlignments

def sentence_probability_wrt_bigram_model(sentence, modelUnigrams, modelBigrams):
    '''
        Given a sentence and a bigram language model, compute the probability of the sentence.
        Parameters
        ----------
        sentence: iterable of words
            The word in the language of the model
            
        modelUnigrams: word-keyed dict of floats
            The probability of each unigram P(w), for a word w in sentence
        
        modelBigrams: word-keyes dict of word-keyed dicts of floats
            The probability P(w2|w1) of a bigram, for a igram (w1,w2) in the sentence.
            It should be P(w2|w1) = modelBigrams[w1][w2]
        
        Returns
        -------
        float:
            The joint probability P(s) under the bigram model for the sentence s.
        
        NOTE: You should normally not need to treat OOV bigrams/unigrams, but if you do,
        (and because the models are deliberately incomplete) it is safe to simply assume the missing
        probability to be 1e-5.
          
    '''
    l = len(sentence)
    logprob = log(modelUnigrams[sentence[0]])
    for i in range(0,l-1):
        logprob += log(modelBigrams.get(sentence[i],dict()).get(sentence[i+1],1e-5))
    return exp(logprob)
    
def rank_valid_translations(sentenceE, m, tEGFNZ):
    '''
        Iterate over all translations, evaluate some score on them and rank them in decreasing probability.
        
        Parameters
        ----------
        sentenceE: an iterable of words
            The source sentence
        
        m: positive integer
            The size of all possible translations

        tFGENZ: a word-keyed dict of word-keyed dicts of probabilities
            The translation probabilities of a foreign token given the source (English) token,
            that only contains entries for the non-zero (NZ) translations.
        
        Returns
        -------
        iterable of
            2-tuple: pair of translation and score
            Return all (valid) alignments scored and ranked.
            e.g.: ( (('the','cat'),0.5), ('the','dog'), 0.4 ), ... )
                     ^           ^  ^    ^           ^   ^ 
                     +-----------+--|----+-----------+---|--------------- Translation
              score --------------- +--------------------+      
    '''
    sentencesF = get_valid_translations(sentenceE, m, tEGFNZ)
    '''######## TODO #########
    Complete this function.
    You can probably do this in 2-3 lines.
    '''
    scoreSentences = []
    sortedScoreSentences = sorted(scoreSentences, key=lambda pair:-pair[1])
    return sortedScoreSentences

def rank_valid_translations_with_model(sentenceE, sentencesF, tNZ, modelUnigramF, modelBigramF):
    '''
        Bonus: You should know what happens here by now.
        Think: What translations should you use?
    '''
    scoreSentences = []
    sortedScoreSentences = sorted(scoreSentences, key=lambda pair:-pair[1])
    return sortedScoreSentences

if __name__ == "__main__":
    ''' ####### Load the dataset ######### '''
    countsE2D = {'the':    dict( (('die', 358343), ('der', 301694), ('den', 112399), ('das', 69253), ('des', 61300), ('dem', 47343)) ),
                 'horse':  dict( (('pferd', 43), ('brunnen', 5), ('ochsen', 4), ('pferdes', 3), ('setzen', 3), ('stöcker', 3), ('pferderennen', 2)) ),
                 'races':  dict( (('rassen', 7), ('völkern', 2), ('rasse', 2), ('rennt', 2), ('pferderennen', 1)) ),
                 'ended':  dict( (('ende', 36), ('beendet', 30), ('zu', 20), ('wurde', 19), ('abgeschlossen', 15), ('endete', 12)) ),
                 'rapidly':dict( (('schnell', 159), ('rasch', 119)) ),
                 # 'NULL':   dict( (('das', 1), ('pferderennen',1), ('pferd',1), ('endete',1), ('rennt',1), ('rasch',1)) )
                 }
    add_null_token(countsE2D)
    unigramsEnglish = {'the': 1085527, 'horse': 99, 'races': 17, 'ended': 328, 'rapidly': 613, 'NULL': 10000 }
    wordsEnglish = 16052703
    
    unigramsForeign = {'kuhhandel': 21, 'dem': 72293, 'ochsen': 5, 'rasse': 83, 'endete': 31, 'man': 24344, 'wurde': 21050, 'stöcker': 4, 'im': 87770, 'pferderennen': 2, 'ende': 3958, 'rasch': 1090, 'des': 100594, 'der': 491791, 'schnell': 2176, 'ein': 72538, 'beendet': 540, 'völker': 1175, 'die': 581109, 'pferdes': 3,
                       'brunnen': 22, 'auf': 102083, 'pferd': 50, 'zu': 176868, 'den': 168228, 'setzen': 2614, 'abgeschlossen': 1139, 'völkern': 311, 'das': 159306, 'rassen': 27, 'rennt': 4,
                       'NULL': 10000
                       }
    bigramsForeign = {'pferderennen': {}, 'im': {'zu': 3, 'das': 1, 'im': 2, 'den': 1, 'dem': 5, 'der': 1, 'ein': 1},'pferd': {'setzen': 1, 'des': 2, 'der': 1}, 'die': {'zu': 1145,
                        'im': 2903, 'rasse': 2, 'das': 1494, 'der': 2292, 'ochsen': 3, 'abgeschlossen': 2, 'setzen': 1, 'den': 1850, 'des': 171, 'ein': 511, 'die': 4773,
                        'rasch': 11, 'ende': 30, 'dem': 614, 'beendet': 1, 'brunnen': 1, 'schnell': 12, 'wurde': 1}, 'rasch': {'zu': 88, 'das': 1, 'den': 1, 'dem': 1,
                        'der': 4, 'die': 26, 'abgeschlossen': 3, 'im': 2, 'ein': 12}, 'ende': {'zu': 248, 'die': 18, 'das': 9, 'setzen': 42, 'dem': 2, 'der': 509, 'im': 3,
                        'den': 4, 'des': 622, 'ein': 11}, 'brunnen': {'zu': 1}, 'schnell': {'zu': 90, 'die': 29, 'den': 6, 'dem': 4, 'der': 4, 'abgeschlossen': 1, 'beendet': 6,
                        'im': 4, 'das': 3, 'ein': 17}, 'völkern': {'zu': 11, 'die': 7, 'im': 1, 'das': 1, 'der': 22, 'ein': 2, 'des': 9}, 'zu': {'zu': 1, 'die': 1, 'den': 7417,
                        'setzen': 902, 'dem': 2848, 'der': 2194, 'rasch': 6, 'ende': 290, 'schnell': 50, 'ein': 18, 'im': 3}, 'das': {'zu': 737, 'das': 142, 'setzen': 8,
                        'der': 579, 'endete': 2, 'ende': 274, 'ein': 431, 'rasch': 6, 'pferd': 27, 'die': 1243, 'den': 331, 'dem': 158, 'des': 33, 'beendet': 2, 'schnell': 12,
                        'wurde': 324, 'im': 588}, 'setzen': {'zu': 18, 'die': 28, 'den': 4, 'dem': 1, 'das': 3, 'im': 3, 'ein': 2}, 'endete': {'die': 1, 'im': 3, 'das': 2},
                        'der': {'zu': 372, 'im': 1031, 'schnell': 3, 'das': 429, 'der': 428, 'rasch': 8, 'beendet': 1, 'rassen': 4, 'ein': 233, 'des': 93, 'die': 2221,
                        'den': 496, 'dem': 245, 'ende': 20, 'rasse': 41, 'wurde': 7}, 'ochsen': {}, 'abgeschlossen': {'zu': 4, 'wurde': 21}, 'rassen': {'das': 1, 'den': 1},
                        'pferdes': {}, 'ein': {'zu': 74, 'das': 1, 'im': 28, 'dem': 7, 'der': 9, 'die': 4, 'ende': 566, 'pferd': 6, 'den': 7}, 'des': {'zu': 21, 'die': 3,
                        'den': 5, 'dem': 3, 'der': 9, 'pferdes': 1, 'ende': 3, 'im': 71, 'wurde': 1, 'des': 1}, 'wurde': {'zu': 89, 'das': 272, 'der': 449, 'ende': 9,
                        'im': 326, 'des': 2, 'ein': 362, 'die': 624, 'den': 30, 'dem': 64, 'abgeschlossen': 2, 'schnell': 1}, 'den': {'zu': 86, 'das': 168, 'im': 336,
                        'der': 295, 'ochsen': 1, 'ende': 5, 'rassen': 6, 'ein': 18, 'des': 25, 'die': 708, 'den': 13, 'dem': 16, 'brunnen': 10, 'schnell': 1, 'völkern': 159},
                        'dem': {'zu': 148, 'im': 129, 'setzen': 1, 'der': 446, 'ende': 133, 'das': 251, 'rasch': 1, 'ein': 159, 'die': 1358, 'den': 24, 'dem': 12, 'des': 49,
                        'wurde': 4}, 'beendet': {'zu': 1, 'die': 2, 'das': 1, 'dem': 1, 'der': 1, 'wurde': 10}, 'rennt': {'der': 2}, 'rasse': {'im': 2, 'des': 1}, 'stöcker': {}
                      }
                         
    wordsForeign = 15257871

    cEGFNZ = invert_dict(countsE2D)
    tEGFNZ = normalize_counts(cEGFNZ, unigramsForeign)
    tFGENZ = normalize_counts(countsE2D, unigramsEnglish)
    
    sentenceE1 = ('the', 'horse', 'races', 'rapidly')
    sentenceF1 = ('pferd', 'die', 'schnell', 'rennt')
    
    sentenceE2 = ('the', 'horse', 'races', 'ended')
    sentenceF2a = ('das', 'pferderennen', 'endete')
    sentenceF2b = ('die', 'pferderennen', 'endete')
    sentenceF2c = ('pferd', 'pferd', 'pferd')
    
    score = sentence_score_f_given_e(sentenceE2, sentenceF2a, tFGENZ)
    score_fast = sentence_score_f_given_e_fast(sentenceE2, sentenceF2a, tFGENZ)
    print("Probability ratio of the two implementations (score/scoreFast): {0}".format(score/score_fast))
    
    print('Ranking alignments ({0})<->({1}):'.format(' '.join(sentenceE2), ' '.join(sentenceF2a)))
    rankedAlignmentsE2F2a = rank_valid_alignments(sentenceE2, sentenceF2a, tFGENZ)
    print_ranking(rankedAlignmentsE2F2a, 5)
    
    m = 3
    print('Ranking translations for m={0} for sentence ({1}):'.format(m, ' '.join(sentenceE2)))
    rankedTranslationsE2 = rank_valid_translations(sentenceE2, m, tFGENZ)
    print_ranking(rankedTranslationsE2, 5)

    '''########### TODO: ######
    Fill the correct models.
    '''
    modelUnigrams = dict()
    modelBigrams = dict()
    print('Ranking language model probabilities:')
    rankingSentences = tuple( (sentence,sentence_probability_wrt_bigram_model(sentence, modelUnigrams, modelBigrams))
                              for sentence in (sentenceF2b, sentenceF2a, sentenceF2c) )
    print_ranking(rankingSentences, k=None)

    bonusReady = False
    if bonusReady:
        # Bonus questions    
        m=4; k=5; sentenceE = sentenceE1
        print('Ranking translations for m={0} with Bigram Lang. Model for sentence ({1}):'.format(m, ' '.join(sentenceE)))
        sentencesF = get_valid_translations(sentenceE, m, tFGENZ)
        # rankedSentencesE1Mdl = rank_valid_translations_with_model(sentenceE, sentencesF, ... , modelUnigrams, modelBigrams)
        print_ranking(rankedSentencesE1Mdl, k=k)
        
        m=3; k=5; sentenceE = sentenceE2
        print('Ranking translations for m={0} with Bigram Lang. Model for sentence ({1}):'.format(m, ' '.join(sentenceE)))
        sentencesF = get_valid_translations(sentenceE, m, tFGENZ)
        # rankedSentencesE1Mdl = rank_valid_translations_with_model(sentenceE, sentencesF, .... , modelUnigrams, modelBigrams)
        print_ranking(rankedSentencesE1Mdl, k=k)
    
    
    
