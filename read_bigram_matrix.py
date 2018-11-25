import re
import numpy as np
import pickle
from sklearn.decomposition import PCA


def read_bigram_matrix(corpus_filename, freqdist_filename, out_filename, word_num=None, stopwords=[]):
    """
    Read bigram model from a corpus and write it to a file.
    Model is saved as a dict with words as keys and vectors of probabilities as entries.
    The order of probabilities in the vector is given by the order of words 
    in the frequency distribution file.
    So, the first entry of the vector for the key "cat", is the probability of seeing "cat"
    after the most common word.
    Also saves the order of words, to make lookup of individual bigram probabilities possible.

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
    stopwords : [String]
        words to ignore
    """
    matrix = dict()
    vector_order = dict()

    most_common = set()
    
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

            #if word in stopwords:
            #    continue
            
            matrix[word] = None
            vector_order[word] = i
            i += 1

            #if i < 5000:
            #    most_common.add(word)

        if word_num is None:
            word_num = i


    vector_order[0] = i #use 0 and 1 as sentence beginning and end symbols
    vector_order[1] = i + 1
    matrix[0] = None
    matrix[1] = None

    word_num += 2
            
    for word in matrix:
        matrix[word] = np.zeros(word_num)
    total_vector = np.zeros(word_num)


    with open(corpus_filename) as corpus_file:
        for i, line in enumerate(corpus_file):
            #if i % 100000 == 0:
            #    print(i)
            words = line.split()
            prev_word = 0 #use 0 as sentence beginning symbol

            for word in words:
                word = word.lower()
                if not word in matrix:
                    continue
                if prev_word is None:
                    prev_word = word
                    continue

                vec_pos = vector_order[prev_word]
                matrix[word][vec_pos] += 1
                total_vector[vec_pos] += 1

                if not word in most_common:#stopwords:
                    prev_word = word

            vec_pos = vector_order[prev_word]
            matrix[1][vec_pos] += 1
            total_vector[vec_pos] += 1
            
                

    for word in matrix:
        for i, total in enumerate(total_vector):
            if total != 0:
                matrix[word][i] = matrix[word][i] / total

    with open(out_filename, "wb") as outfile:
        pickle.dump(matrix, outfile)
    order_filename = out_filename + "_order"
    with open(order_filename, "wb") as orderfile:
        pickle.dump(vector_order, orderfile)


def pca_matrix(input_filename, output_filename, n_components=300):

    with open(input_filename, "rb") as input_file:
        matrix_dict = pickle.load(input_file)

    vector_order_filename = input_filename + "_order"
    with open(vector_order_filename, "rb") as vector_order_file:
        vector_order = pickle.load(vector_order_file)

    word_num = len(vector_order)
    print(vector_order[1])
    print(vector_order[0])
    matrix = np.empty((word_num, word_num), float)

    for word in matrix_dict:
        word_pos = vector_order[word]
        matrix[word_pos] = matrix_dict[word]

    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    transformed_matrix = pca.transform(matrix)

    transformed_dict = dict()

    for word in vector_order:
        word_pos = vector_order[word]
        transformed_dict[word] = transformed_matrix[word_pos]

    with open(output_filename, "wb") as output_file:
        pickle.dump(transformed_dict, output_file)
    


            

stopwords = [
    "", "(", ")", "a", "about", "an", "and", "are", "around", "as", "at",
    "away", "be", "become", "became", "been", "being", "by", "did", "do",
    "does", "during", "each", "for", "from", "get", "have", "has", "had", "he",
    "her", "his", "how", "i", "if", "in", "is", "it", "its", "made", "make",
    "many", "most", "not", "of", "on", "or", "s", "she", "some", "that", "the",
    "their", "there", "this", "these", "those", "to", "under", "was", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your"
]
        
#read_bigram_matrix("../bnc_sentences", "../bnc_freqdist_lowercase.txt", "../bigram_matrix_10k_new_stopwords.pkl", word_num=10000, stopwords=stopwords)
read_bigram_matrix("../bnc_sentences", "../bnc_freqdist_lowercase.txt", "../bigram_matrix_10k.pkl", word_num=10000, stopwords=[])
        
        
