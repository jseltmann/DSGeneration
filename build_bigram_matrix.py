import scipy.sparse
import numpy as np
import pickle
import sys

def build_ngram_progability_matrix(corpus, freq_dist_file, num_words=10000):

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
    
    #freq_dict = {i[1]:0 for i in freqs_from_file[:num_words]} 
    #freq_dict = {i:0 for i in range(num_words)} 


    print("calculating number of sentences...")

    # This returns the number of sentences, i.e: the frequency of START$_and END$_
    howmanysents = len([i for i in open(corpus)])
    print("number of sentences: ", howmanysents, "\n")

    # These two lines add the types for start and end of sentence to the dictionary with their frequency
    #freq_dict["START$_"] = 0#howmanysents
    #freq_dict["END$_"] = 0#howmanysents

    # this returns the indices of the matrix
    print("building frequency list dictionary...")
    #vocab_order = {w:i for i,w in enumerate(freq_dict.keys())}
    vocab_order = {w[1]:i for i,w in enumerate(freqs_from_file[:num_words])}
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
                matrix[vocab_order[i[1]], vocab_order[i[0]]] += 1
                freq_dict[vocab_order[i[0]]] += 1
                non_zero_positions.add((vocab_order[i[1]], vocab_order[i[0]]))

                unigram_counts[i[1]] += 1

                if i[0] == "START$_":
                    unigram_counts[i[0]] += 1

        
        # this is just a counter to give the user some feedback
        count += 1
        if count%10000 == 0:
            print(count, " sentences processed ...")

    print("counted bigrams ...")

    #normalize the counts
    count = 0
    for row, col in non_zero_positions:
        matrix[row, col] /= freq_dict[col]

        if count % 5000 == 0:
            print(count)
        count += 1


    #calculate unigram probabilities
    print("calculating unigram probabilities ...")
    count_sum = sum(unigram_counts.values())

    for w in unigram_counts:
        unigram_counts[w] /= count_sum
    
        
    return matrix, unigram_counts, vocab_order



foo, baz, bar =  build_ngram_progability_matrix(sys.argv[1], sys.argv[2], num_words=int(sys.argv[3]))


print ("saving the matrix file as " + sys.argv[1][:-4]+"_matrix.pkl")
pickle.dump(foo, open(sys.argv[1][:-4]+"_matrix.pkl", "wb"))

print("saving the vector index file as " + sys.argv[2][:-4]+"_vector_index.pkl")
pickle.dump(bar, open(sys.argv[2][:-4]+"_vector_index.pkl", "wb"))

print("saving the unigram index file as " + sys.argv[1][:-4]+"_unigram_probs.pkl")
pickle.dump(baz, open(sys.argv[1][:-4]+"_unigram_probs.pkl", "wb"))


