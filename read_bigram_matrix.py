import re
import numpy as np
import nltk
import pickle


def read_bigram_matrix(corpus_filename, freqdist_filename, out_filename, word_num=None):
    """
    Read bigram model from a corpus and write it to a file.
    Model is saved as a dict with words as keys and vectors of probabilities as entries.
    The order of probabilities in the vector is given by the order of words 
    in the frequency distribution file.
    So, the first entry of the vector for the key "cat", is the probability of seeing "cat"
    after the most common word.
    Also saves the order of words, to make lookup of individual bigram probabilities possible-

    Parameters
    ----------
    corpus_filename : String
        Filename of the file containing the corpus.
    freqdist_filename : String
        Filename of the file containing the frequency distribution for the corpus.
        Assumes that each line of the file containes a number and a word
        and that the words are in descending order according to the frequency.
    out_filename : String
        Filename of the file in which to store the model.
    word_num : Int
        Number of words to use. 
        If it is None, all the words in the frequency distribution file are used.
    """
    matrix = dict()
    vector_order = dict()

    with open(freqdist_filename) as freqdist_file:
        i = 0
        reg = r"\s*[0-9]+\s(.+)"

        for line in freqdist_file:
            if word_num and i == word_num:
                break
            m = re.match(reg, line)
            if not m:
                print(line)
                continue
            word = m.group(1).lower()
            matrix[word] = None
            vector_order[word] = i
            i += 1

        if word_num is None:
            word_num = i

    for word in matrix:
        matrix[word] = np.zeros(word_num)


    with open(corpus_filename) as corpus_file:
        for i, line in enumerate(corpus_file):
            #if i % 100 == 0:
            #    print(i)
            words = line.split()
            prev_word = None

            for word in words:
                if not word in matrix:
                    continue
                if prev_word is None:
                    prev_word = word
                    continue

                vec_pos = vector_order[prev_word]
                matrix[word][vec_pos] += 1

                prev_word = word

    for word in matrix:
        total = sum(matrix[word])
        if total > 0:
            matrix[word] = matrix[word] / total

    with open(out_filename, "wb") as outfile:
        pickle.dump(matrix, outfile)
    order_filename = out_filename + "_order"
    with open(order_filename, "wb") as orderfile:
        pickle.dump(vector_order, orderfile)


            

        
read_bigram_matrix("../bnc_sentences", "../bnc_freqdist_lowercase.txt", "../bigram_matrix_10000.pkl", word_num=10000)
        

        
