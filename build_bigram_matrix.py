import scipy.sparse
import numpy as np
import pickle
import sys
import os.path

def build_ngram_probability_matrix(corpus, freq_dist_file, matrix_filename, num_words=10000, stopwords=[]):

    """
    The function returns a scipy.sparse.lil_matrix() object containing bigram probabilities in a given corpus and
    a vector index file as a dictionary. 
    
    In order to make it work you need to provide:
    - the path to  a corpus (pre-trokenized)
    - the list of frequencies of the same corpus. 
    - the maximum number of types that you want in your lexicon

    In order to obtain the frequencies, a fast way is to run the follwing terminal command (assuming that the tokens are separated by a spacing character like \t or space):
    
    tr "\t" "\n" < corpus_file | sort | uniq -c | sort -nr > frequency_file
    
    """




    print("reading the frequency file ....\n")
    freqs_from_file = [tuple(i.split()) for i in open(freq_dist_file)]
    
    # we take only the first 100,000 words
    # Note: here a very bizarre things happens if I don't restrict the number of words: some of the values are randomly reduced
    # to 1, like "in" or "that"
    if num_words > 100000:
        num_words = 100000
    


    print("calculating number of sentences...")

    # This returns the number of sentences, i.e: the frequency of START$_and END$_
    howmanysents = len([i for i in open(corpus)])
    print("number of sentences: ", howmanysents, "\n")

    # this returns the indices of the matrix
    print("building frequency list dictionary...")
    vocab_order = {w[1]:i for i,w in enumerate(freqs_from_file[:num_words])}
    # add tokens for sentence beginning and end
    vocab_order["START$_"] = len(vocab_order)
    vocab_order["END$_"] = len(vocab_order)
    print("number of types (including START$_ and END$_ ):", len(vocab_order), "\n")

    freq_dict = {i:0 for i in range(len(vocab_order))}

    unigram_counts = {w:0 for w in vocab_order}
    
    
    # the empty matrix is instantiated, the shape of it is equal to the brigram lexicon
    matrix = scipy.sparse.lil_matrix((len(freq_dict), len(freq_dict)), dtype=np.float64)
    
    count = 0
    non_zero_positions = set()
    for sent in open(corpus):

        # every line in the corpus file is transformed in a list
        current_sent = sent.strip().split()

        # we extend the list of tokens by adding sentence start and end strings
        current_sent = ["START$_"] + current_sent + ["END$_"]

        # we create the list of brigrams (as tuples) form the sentence 
        bigrams =[(current_sent[i], current_sent[i+1]) for i in range(len(current_sent)-1)]
        
        # if both words in the bigram are in the lexicon we add the count to the matrix
        for i in bigrams:
            if i[0] in vocab_order and i[1] in vocab_order:
                if i[0] in stopwords:
                    continue
                matrix[vocab_order[i[1]], vocab_order[i[0]]] += 1
                freq_dict[vocab_order[i[0]]] += 1
                non_zero_positions.add((vocab_order[i[1]], vocab_order[i[0]]))

                unigram_counts[i[1]] += 1

                if i[0] == "START$_":
                    unigram_counts[i[0]] += 1


        
        # this is just a counter to give the user some feedback
        count += 1
        if count%100000 == 0:
            print(count, " sentences processed ...")

    print("calculating bigram probabilities ...")
    #normalize the counts
    count = 0
    for row, col in non_zero_positions:
        matrix[row, col] /= freq_dict[col]

        if count % 500000 == 0:
            print(count, "bigrams processed ...")
        count += 1


    #calculate unigram probabilities
    print("calculating unigram probabilities ...")
    count_sum = sum(unigram_counts.values())

    for w in unigram_counts:
        unigram_counts[w] /= count_sum
    
    with open(matrix_filename, "wb") as matrix_file:
        pickle.dump((matrix, unigram_counts, vocab_order), matrix_file)
        



def read_predict_vectors(predict_filename, matrix_directory, prev_matrix_name=None, word_num=None):
    """
    Read predict vectors from http://clic.cimec.unitn.it/composes/semantic-vectors.html
    into matrix. Assigns equal probabilities to the unigrams.

    Parameters
    ----------
    predict_filename : str
         File containing the predict vectors.
    matrix_filename : str
         File to save the matrix to.
    prev_matrix_name : str
         Filename of an existing matrix. 
         Use to have this matrix have the same vocabulary and word order as the old one.
    word_num : int
         Number of words to include in the matrix.
         If None, use all words.
         If prev_matrix is given, word_num is ignored.
    """

    vocab_order = dict()
    unigram_probs = dict()

    if prev_matrix_name is not None:
        with open(prev_matrix_name, "rb") as prev_matrix_file:
            old_vocab_order = pickle.load(prev_matrix_file)[2]
        word_num = len(old_vocab_order)
    else:
        old_vocab_order = None

    if word_num is None:
        word_num = 0
        with open(predict_filename) as predict_file:
            for line in predict_file:
                vec_len = len(line.split("\t")) - 1
                word_num += 1
    else:
        with open(predict_filename) as predict_file:
            for line in predict_file:
                vec_len = len(line.split("\t")) - 1
                break

    
    matrix = scipy.sparse.lil_matrix((word_num, vec_len), dtype=np.float64)
        

    with open(predict_filename) as predict_file:
        for i, line in enumerate(predict_file):
            entries = line.split("\t")
            word = entries[0].lower()
            try:
                vector = list(map(float, entries[1:]))
            except Exception as e:
                print(i)
                print(line)
                continue

            if old_vocab_order is not None:
                if not word in old_vocab_order:
                    continue
                pos = old_vocab_order[word]
            else:
                if i == word_num:
                    break
                pos = i
            for j in range(vec_len):
                matrix[pos,j] = vector[j]
            vocab_order[word] = pos
            unigram_probs[word] = 1

    num_words = len(unigram_probs)
    for word in unigram_probs:
        unigram_probs[word] = 1 / num_words

    matrix.tocsc()

    with open(matrix_filename, "wb") as matrix_file:
        pickle.dump((matrix, unigram_probs, vocab_order), matrix_file)


    



