import pickle
import gmpy2
from matrix_class import DS_matrix
import numpy as np
from collections import defaultdict




def perplexity(corpus_filename, model_filename):
    """
    Calculate perplexity of a bigram model on a test corpus.

    Parameters
    ----------
    corpus_filename : string
        Filename of the test corpus.
        Assumes that each line of the corpus contains one word-tokenized sentence.
    model_filename : string
        Filename of the bigram matrix.

    Return
    ------
    perplexity : float
        Perplexity of the model on the corpus.
    """

    model = DS_matrix(model_filename)


    bigram_counts = np.zeros((len(model.vector_order),len(model.vector_order)), dtype=np.int64)
    unigram_counts = np.zeros(len(model.vector_order), dtype=np.int64)

    with open(corpus_filename) as corpus:
        counter = 0

        for i, line in enumerate(corpus):
            prev_word = 0
            words = line.split()
            for word in words:
                if not model.contains(word):
                    prev_word = None
                    continue
                if prev_word is None:
                    pos = model.vector_order[word]
                    unigram_counts[pos] += 1
                else:
                    pos1 = model.vector_order[prev_word]
                    pos2 = model.vector_order[word]
                    bigram_counts[pos1][pos2] += 1
                prev_word = word
                counter += 1

            word = 1
            if prev_word is None:
                pos = model.vector_order[word]
                unigram_counts[pos] += 1
            else:
                pos1 = model.vector_order[prev_word]
                pos2 = model.vector_order[word]
                bigram_counts[pos1][pos2] += 1

    perplexity = 1
    for w1 in model.vector_order:
        for w2 in model.vector_order:
                
            pos1 = model.vector_order[w1]
            pos2 = model.vector_order[w2]
            bigram_count = bigram_counts[pos1][pos2]

            if bigram_count != 0:
                prob = model.get_bigram_prob(w2, w1)
                if prob == 0:
                    #backoff
                    prob = model.get_unigram_prob(w2)
                for _ in range(bigram_count):
                    perplexity *= ((1/prob) ** (1/counter))

    for w in model.vector_order:
        pos = model.vector_order[w]
        unigram_count = unigram_counts[pos]

        if unigram_count != 0:
            prob = model.get_unigram_prob(w)
            for _ in range(unigram_count):
                perplexity *= ((1/prob) ** (1/counter))

    return perplexity


        
p = perplexity("../bnc_sentences_lowercase_test.txt", "../bnc_sentences_lowercase_train_matrix.pkl")
print(p)
            
