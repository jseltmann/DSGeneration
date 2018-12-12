import pickle
from matrix_class import DS_matrix
import numpy as np
from collections import defaultdict
import scipy.sparse




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


    bigram_counts = scipy.sparse.lil_matrix((len(model.vocab_order),len(model.vocab_order)), dtype=np.int64)
    #unigram_counts = scipy.sparse.lil_matrix((1,len(model.vocab_order)), dtype=np.int64)
    unigram_counts = np.zeros(len(model.vocab_order), dtype=np.int64)

    with open(corpus_filename) as corpus:
        counter = 0

        for i, line in enumerate(corpus):
            prev_word = "START$_"
            words = line.split()
            for word in words:
                if not model.contains(word):
                    prev_word = None
                    continue
                if prev_word is None:
                    pos = model.vocab_order[word]
                    unigram_counts[pos] += 1
                else:
                    pos1 = model.vocab_order[prev_word]
                    pos2 = model.vocab_order[word]
                    #print("pos1", pos1)
                    #print("pos2", pos2)
                    #print("check", bigram_counts._check_row_bounds(pos1))
                    if bigram_counts[pos1, pos2] == 0:
                        bigram_counts[pos1, pos2] = 1
                    else:
                        bigram_counts[pos1, pos2] += 1
                prev_word = word
                counter += 1

            word = "END$_"
            if prev_word is None:
                pos = model.vocab_order[word]
                unigram_counts[pos] += 1
            else:
                pos1 = model.vocab_order[prev_word]
                pos2 = model.vocab_order[word]
                if bigram_counts[pos1, pos2] == 0:
                    bigram_counts[pos1, pos2] = 1
                else:
                    bigram_counts[pos1, pos2] += 1

    print("counter", counter)
    perplexity = 1
    for i, w1 in enumerate(model.vocab_order):
        if i % 2500 == 0:
            print(i, "words processed ...")
        for w2 in model.vocab_order:
                
            pos1 = model.vocab_order[w1]
            pos2 = model.vocab_order[w2]
            bigram_count = bigram_counts[pos1, pos2]

            if bigram_count != 0:
                prob = model.get_bigram_prob(w2, w1)
                if prob == 0:
                    #backoff
                    prob = model.get_unigram_prob(w2)
                if prob == 0:
                    continue
                for _ in range(bigram_count):
                    perplexity *= ((1/prob) ** (1/counter))

    for w in model.vocab_order:
        pos = model.vocab_order[w]
        unigram_count = unigram_counts[pos]

        if unigram_count != 0:
            prob = model.get_unigram_prob(w)
            for _ in range(unigram_count):
                perplexity *= ((1/prob) ** (1/counter))

    return perplexity


        
p = perplexity("../generated_sentences.txt", "../matrix_50k/_matrix.pkl")
print("perplexity on bigram sentences",p)


            
