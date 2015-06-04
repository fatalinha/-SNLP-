'''
Created on May 27, 2015

@author: janis
'''

from collections import Counter

def read_tagged_file(file, column_separator = '\t'):
    '''
    Returns a tuple of n-tuples.
    Each n-tuple has an entry for each column, separated by the specified separator.
    
    Example:
        The text
            """
            So    ADV
            far    ADV
            these    DET
            """
        yields:
            (('So','ADV'), ('far',ADV'), ('these,'DET'))
    '''
    with open(file,'r') as fid:
        text = fid.read().lower()
    taggedWords = tuple( tuple(line.split(column_separator)) for line in text.split('\n') )
    # Remove last entry if empty
    if(len(taggedWords[-1][-1])==0): taggedWords=taggedWords[:-1]
    return taggedWords
    
def count_transitions(taggedWords):
    """ 
    Count transition occurrences. This process is done once per training corpus;
    the resulting data are meant to be used for estimating transmission probabilities during
    prediction time.
    
    Parameters
    ------------
    taggedWords : a sequence of (word,tag) tuples
        The training corpus
    
    Returns
    ------------
    tuple
        This can be any data you might need during the probability estimation.
        One example would be a tuple of 2 dictionaries:
            countTransitionsOfPostags: A POS-keyed dictionary of POS-keyed dictionaries
                e.g. could store the conditional counts such that
                countTransitionsOfPostags[POS1][POS2] counts the number of
                transmissions from POS1 to POS2  
            countPostagsPerPostag: A POS-keyed dictionary
                e.g. could count the number of total originating (previous) 
                postag occurrences, such that countPostagsPerPostag[POS1] counts the
                total transitions that originate from a (previous) postag POS1
    """
    
    ''' In case you adopt the above approach, some preliminary steps have already been provided.
    Feel free to modify them apropriately.'''
      
    allPostags = tuple(zip(*taggedWords))[1]
    distinctPostags = set(allPostags)

    # Initialize countTransitionsOfPostags with empty dicts for each tag
    countTransitionsOfPostags = dict((postag, dict()) for postag in distinctPostags)
    bigrams = tuple(zip(allPostags, allPostags[1:]))
    countTransitionsOfPostags = (Counter(bigrams))

    # Initialize countPostags with zero entries
    countPostagsPerPostag = dict( (postag,0) for postag in distinctPostags )
    for postag in distinctPostags:
        countPostagsPerPostag = Counter(allPostags)


    # assert(countTransitionsOfPostags['det']['noun'] == 7898)
    # assert(countPostagsPerPostag['noun'] == 27561)
    
    countDataTransitions = countTransitionsOfPostags, countPostagsPerPostag
    return countDataTransitions

def count_emissions(taggedWords):
    '''
    Count emission occurrences. This process is done once per training corpus;
    The resulting data are meant to be used for estimating emission probabilities during
    prediction time.
    
    Parameters
    ------------
    taggedWords : a sequence of (word,tag) tuples
        The training corpus
    
    Returns
    ------------
    tuple
        This can be any data you might need during the probability estimation.
        One example would be a tuple of 2 dictionaries:
            countEmissionsOfPostags: A POS-keyed dictionary of word-keyed dictionaries
                e.g. could store the conditional counts such that
                countEmissionsOfPostags[POS][W] counts the number of
                emissions of word W when tagged with postag POS  
            countWordsPerPostag: A POS-keyed dictionary
                e.g. could count the number of total postag occurrences
                such that countWordsPerPostag[POS] counts the total emmssions
                that originate from a postag POS
    '''
    allCounts = Counter(taggedWords)
    tokens,postags = tuple(zip(*allCounts.keys()))
    tokens=set(tokens);postags=set(postags)
    # Initialize countEmissionsOfPostags with empty dicts for each word
    countEmissionsOfPostags = dict( (postag,dict()) for postag in postags )
    # Initialize countPostags with zero entries
    ''' TODO: Fill (or re-implement) this function '''
    
    '''
    In case you decide to use the above representation,
    then if you use the training corpus for counting the 
    assertions below might help you in testing your code. '''
    # assert(countEmissionsOfPostags['adv']['so'] == 78)
    # assert(countWordsPerPostag['det'] == 13382)
    # assert(sum(countWordsPerPostag.values()) == 108503)
    
    countDataEmissions = countEmissionsOfPostags, countWordsPerPostag
    return countDataEmissions

def get_transition_probability(previousPostag, countDataTransitions):
    '''
    Computes probability estimates for the transitions.
    
    Parameters:
    --------------
    previousPostag: string
        The POS-tag from which the transition probabilities are requested.
    countDataTransitions: tuple
        The count data computed by function count_transitions. 
    Returns:
    --------------
    dict
        The return type MUST be a dictionary with postags as keys and floats as values.
        The values should be such that probabilityTransitions[POS1][POS2] gives the
        transition probability P(POS2|POS1).
    '''
    
    ''' TODO: Fill (or re-implement) this function '''
    countTransitionsOfPostags, countPostagsPerPostag = countDataTransitions
    
    return probabilityTransitions

def get_emission_probability(postag, tokens, epsilon, countDataEmissions):
    '''
    Computes probability estimates for the emissions.
    
    Parameters:
    --------------
    postag: string
        The POS-tag from which the emission probabilities are requested.
    tokens: sequence of strings
        You must provide emission probabilities for each token in this sequence.
        NOTE: It can happen (and you MUST take this into consideration) that some tokens
        will be OOV words, that means that there will *not* be entries in your count
        dictionaries.
    epsilon: float
        This is the smoothing parameter. Use this in the Lidstone smoothing computation formula.
    countDataEmissions:
        The count data computed by function count_emissions. 
    Returns:
    --------------
    dict
        The return type MUST be a dictionary with ALL the tokens as keys and floats as values.
        The values should be such that probabilityEmissions[POS][TOKEN] gives the
        emission probability P(TOKEN|POS).
    '''
    countEmissionsOfPostags, countWordsPerPostag = countDataEmissions
    N = countWordsPerPostag[postag]
    V = len(tokens)
    denominator = N+epsilon*V
    countEmissionsOfPostag = countEmissionsOfPostags[postag]
    def lidstone_estimate(token):
        count = countEmissionsOfPostag[token] if token in countEmissionsOfPostag else 0
        return (count+epsilon)/denominator
    probabilityEmissions = dict( (token,lidstone_estimate(token)) for token in tokens)
    return probabilityEmissions

def HMM_predict(symbols, get_TP, get_EP, probabilityState0):
    """
    Estimate predictions for the specific symbols, using the HMM described by the 
    transition and emission probabilities.
    
    This function is provided as is and (under normal conditions) you do not need to modify
    its contents.
    
    Parameters:
    -------------
        symbols: sequence
            The sequence of items that need to be tagged (in our case, words)
        get_TP: function
            A function that computes transition probabilities. Called as:
            get_TP(POS) (e.g.: get_TP('det')
            It must yield a dictionary of postags and their probabilities, such that
            get_TP(POS1)[POS2] = P(POS2|POS1)
        get_EP: function
            A function that computes emission probabilities. Called as:
            get_EP(POS) (e.g.: get_EP('det')
            It must yield a dictionary of words and their probabilities, such that
            get_EP(POS)[TOKEN] = P(TOKEN|POS)
        probabilityState0: dict with POS tags as keys and floats as values
            A dictionary of the probabilities of the initial states, such that
            probabilityState0[POS] = P(POS)
    
    Returns:
    ------------
        tuple of (maxStateSymbols,logprob):
            maxStateSymbols: sequence 
                The most probable states (POS tags) as predicted by the HMM model states,
                for the specific observations described y symbols (the words).
            logprob: float (logarithmic, natural base)
                The JOINT probability P(X,Y) of the most likely sequence AND the observations.
    """
    import numpy as np
    import time
    timeStart = time.time()
    message = lambda x: print(x, flush=True, end="") 
    """ This part converts the slow dictionaries to very fast numpy matrices"""
    def create_map(elements):
        symbols = tuple(set(elements))
        mapSymbols2Indices = dict(zip(symbols,range(0,len(symbols))))
        map_to_index = mapSymbols2Indices.__getitem__
        map_to_symbol = symbols.__getitem__
        return {'2sym':map_to_symbol, '2idx':map_to_index, 'N': len(symbols), 'all':symbols}
    
    def compute_matrices(mapSymbols, mapStates):
        A = np.zeros((Q,Q),np.float64)
        B = np.zeros((T,Q),np.float64)
        p0 = np.zeros((Q,1),np.float64)
        for state in mapStates['all']:
            indexState = mapStates['2idx'](state)
            
            transitionsFromState = get_TP(state)
            newStates,probabilities = tuple(zip(*transitionsFromState.items()))
            indicesNewStates = tuple(map(mapStates['2idx'], newStates))
            A[indexState, indicesNewStates] = probabilities
            
            emissionsFromState = get_EP(state, mapSymbols['all'])
            emittedSymbols,probabilities = tuple(zip(*emissionsFromState.items()))
            indicesEmittedWords = tuple(map(mapSymbols['2idx'], emittedSymbols))
            B[indicesEmittedWords, indexState] = probabilities
        
            p0[indexState] = probabilityState0[state]
        return A,B,p0
    # Map inputs to indices
    mapSymbols = create_map(symbols)
    mapStates = create_map(probabilityState0.keys())
    Q = mapStates['N']
    T = len(symbols)
    
    message('HMM Predict: Creating matrices... ')
    A,B,p0 = compute_matrices(mapSymbols, mapStates)
    
    message('Mapping inputs... ')
    v = np.fromiter(map(mapSymbols['2idx'],symbols),np.int)

    backtrack = np.zeros((Q,T), np.int);
    
    message('Running Viterbi (FW variant)...')
    mu = np.log(p0*B[v[0],:,None]);
    for t in range(1,T):
        mutmp = np.log(A) + np.log(B[v[t],:,None]) + mu.T;
        mu = np.max(mutmp,axis=1)
        backtrack[:,t-1] = np.argmax(mutmp,axis=1)
    
    message('Backtracking...')
    logprob = np.max(mu)
    maxstate = [np.argmax(mu)];
    for t in range(T-2,-1,-1):
        maxstate = [backtrack[maxstate[0],t]]+maxstate
    
    message('Unmapping integers...')
    maxStateSymbols = tuple(map(mapStates['2sym'],maxstate))
    message('Done ({0:5.4} seconds)\n'.format(time.time()-timeStart))
    return maxStateSymbols,logprob

def evaluate(tagsGT, tagsPred, words, OOV, countSentences = False):
    """
    Evaluate the prediction in terms of MCR, ignoring dot tags.
    
    Parameters:
    --------------
        tagsGT: sequence of POS tags
            The ground truth/Golden standard (correct tags)
        tagsPred: sequence of POS tags
            The predicted tags.
        words: sequence of words
            The words of the text that was tagged
        OOV: set of tokens
            The tokens that are considered OOV.
    
    Returns:
    --------------
        tuple of MCR, MCRRatioOOV, mistakenSentences
            MCR: float
                The missclassification rate. In this computation,
                wrong dot tags count as a missclassification, whereas a correctly
                tagged dot tag does not count as a correct classification.
            MCRRatioOOV: float
                The percentage of istakes that are due to OOV words.
            mistakenSentences: sequence of strings
                A list of missclassified tags within their sentences.
    """
    numWords = len(tagsGT)
    assert(numWords == len(tagsPred))
    dots = missclassifications = missclsOOV = 0
    
    mistakenSentences = []
    for ind in range(0,numWords):
        if tagsGT[ind] != tagsPred[ind]:
            missclassifications += 1
            if words[ind] in OOV:
                missclsOOV += 1
            if countSentences:
                end = next(i for i in range(ind,len(words)) if tagsGT[i] == '.')
                try:
                    start = next(i for i in range(ind,-1,-1) if tagsGT[i] == '.')+1
                except StopIteration:
                    start = 1
                sent = list(words[start-1:end+1])
                sent[ind-start+1] = '*'+sent[ind-start+1]+'*'
                mistake = ' '.join(sent)
                mistakenSentences.append(mistake)
        else:
            if tagsGT[ind] == '.':
                dots += 1
    # Dots are too easy to count as correct
    MCR = missclassifications / (numWords - dots)
    MCRRatioOOV = missclsOOV/missclassifications
    return MCR, MCRRatioOOV, mistakenSentences

if __name__ == '__main__':
    import numpy as np

    taggedWords = {
                   'train': read_tagged_file('brown-train.txt'),
                   'development': read_tagged_file('brown-development.txt'),
                   'test': read_tagged_file('brown-test.txt')
                  }

    trainTokens = set(tuple(zip(*taggedWords['train']))[0])
    
    print('Training HMM: Counting Transitions')
    countDataTransitions = count_transitions(taggedWords['train'])
    print('Training HMM: Counting Emissions')
    countDataEmissions = count_emissions(taggedWords['train'])

    get_TP = lambda postag: get_transition_probability(postag, countDataTransitions) 
    get_EP_epsilon = lambda postag, words, epsilon: get_emission_probability(postag, words, epsilon, countDataEmissions) 
    
    # Some tests of probability estimates. Use as a guideline if you want
    assert(get_TP('det')['noun'] == 7898 / 13382 )
    assert(get_EP_epsilon('det', ('the',), 0)['the'] == 7470 / 13382)
    # Test summation to unit
    assert(abs(sum(get_EP_epsilon('det', trainTokens.union(('OOV',)), 1).values())-1) < 1e-8)
    
    print('Training HMM: Computing Initial State Probabilities')
    probabilityPostag0 = get_TP('.')
    
    testWords,testPostags = tuple(zip(*taggedWords['test']))
    OOVtest = set(testWords).difference(trainTokens)

    
    def evaluate_HMM_for_epsilon(evaluateWords, evaluatePostags, OOVWords, epsilon, countSentences = False):
        get_EP = lambda postag, tokens: get_EP_epsilon(postag,tokens, epsilon)
        print('Testing HMM: Tagging started (epsilon: {0})'.format(epsilon))
        predictedPostags, logProbability = HMM_predict(evaluateWords, get_TP, get_EP, probabilityPostag0)
        MCR, MCRRatioOOV, sentences = evaluate(evaluatePostags, predictedPostags, evaluateWords, OOVWords, countSentences)
        print('Testing HMM: MCR={0:5.4}, out of which {1:5.3} due to OOV words.'.format(MCR, MCRRatioOOV))
        return MCR, MCRRatioOOV, sentences, logProbability
    
    ''' TODO: find/propose a good value for epsilon '''
    epsilonBest = 1e-4
    
    plottingEnabled = True
    if plottingEnabled:
        def evaluate_epsilon(epsilon):
            ''' Fill this function '''
            MCR = 0
            return MCR

        plotEpsilon = np.logspace(-6,0,20)
        plotMCR = tuple(map(evaluate_epsilon, plotEpsilon))
    
        plt.semilogx(plotEpsilon,plotMCR,'b.-',label='MCR');
        plt.xlabel('Epsilon')
        plt.ylabel('MCR')
        plt.title('MCR for various epsilon')
        plt.savefig('epsilonMCR.pdf')
    
    MCR, MCRRatioOOV, sentences, logProbability = evaluate_HMM_for_epsilon(testWords, testPostags, OOVtest, epsilonBest, countSentences = True)
    print('Example mistakes:\n\t' + '\n\t'.join(sentences[0:30:3]))
    
