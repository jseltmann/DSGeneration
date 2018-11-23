import pickle
import gmpy2
from matrix_class import DS_matrix




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

    order_path = model_filename + "_order"

    model = DS_matrix(model_filename, order_path=order_path)

    counter = 0
    with open(corpus_filename) as corpus:
        for line in corpus:
            words = line.split()
            for word in words:
                if model.contains(word):
                    counter += 1

            counter += 1

    perplexity = 1
            
    with open(corpus_filename) as corpus:
        for i, line in enumerate(corpus):
            if i % 1000 == 0:
                print(i)
            words = line.split()

            prev_word = 0
            for word in words:

                if not model.contains(word):
                    prev_word = None
                    continue
                
                if prev_word is None:
                    prob = model.get_unigram_prob(word)
                else:
                    prob = model.get_bigram_prob(word, prev_word)
                    if prob == 0:
                        prob = model.get_unigram_prob(word)

                perplexity *= (1/prob) ** (1/counter)

            if prev_word is None:
                prob = model.get_unigram_prob(1)
            else:
                prob = model.get_bigram_prob(1, prev_word)
                if prob == 0:
                    prob = model.get_unigram_prob(1)
            perplexity *= (1/prob) ** (1/counter)

    return perplexity


        
p = perplexity("../bnc_sentences", "../bigram_matrix_10k.pkl")
print(p)
            
