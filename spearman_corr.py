import re
import sys
import pickle
from scipy.stats.mstats import spearmanr
from scipy.spatial.distance import cosine

from matrix_class import DS_matrix


def calculate_spearman(gold_filename, matrix_filename, similarity_function):
    """
    Calculate Spearman coefficient between a corpus of similarities of word pairs
    and a bigram model as produced by read_bigram_matrix.py.

    Parameters
    ----------
    gold_filename : String
        Filename of the corpus.
        Assumes that there is one word pair per line
        in the format "word1 word2 similarity"
    matrix_filename : String
        File containg the bigram model.
    unigram_filename : String
        File containing the unigram probabilities.
    vocab_order_filename : String
        File containing the order of the vectors in the matrix.
    similarity_function : function
        Function for calculating influence of two vectors.

    Return
    ------
    spearman : Float
        Spearman coefficient
    """


    reg = r"(\S+)\s(\S+)\s(\S+)"

    gold_list = []

    with open(gold_filename) as gold_file:

        for line in gold_file:
            m = re.match(reg, line)
            if not m:
                print(line)
                continue

            word1 = m.group(1)
            word2 = m.group(2)
            sim = m.group(3)

            gold_list.append((word1,word2,sim))

    matrix_similarity_list = []
    gold_similarity_list = []
    

    matrix = DS_matrix(matrix_filename)
    
    for word1, word2, sim in gold_list:
        if not matrix.contains(word1) or not matrix.contains(word2):
            continue

        similarity = similarity_function(matrix.get_vector(word1), matrix.get_vector(word2))
        matrix_similarity_list.append(similarity)

        gold_similarity_list.append(float(sim))

    spearman, _ = spearmanr(matrix_similarity_list, gold_similarity_list)

    return spearman




spearman = calculate_spearman(sys.argv[1], sys.argv[2], lambda x,y : 1 - cosine(x,y))
print("spearman correlation cosine similarity:", spearman)
