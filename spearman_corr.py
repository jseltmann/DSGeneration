import re
import pickle
from scipy.stats.mstats import spearmanr
from scipy.spatial.distance import cosine


def calculate_spearman(gold_filename, matrix_filename):
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
    
    with open(matrix_filename, "rb") as matrix_file:
        matrix = pickle.load(matrix_file)

        for word1, word2, sim in gold_list:
            #word1, word2, sim = entry
            if not word1 in matrix or not word2 in matrix:
                continue

            similarity = 1 - cosine(matrix[word1], matrix[word2])
            matrix_similarity_list.append(similarity)

            gold_similarity_list.append(float(sim))

    spearman, _ = spearmanr(matrix_similarity_list, gold_similarity_list)

    return spearman


    
            




spearman = calculate_spearman("../MEN/MEN_dataset_natural_form_full", "../rev_matrices/bigram_matrix_10000.pkl")            
print("spearman coefficient:")
print(spearman)

