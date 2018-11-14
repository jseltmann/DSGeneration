import re
#import pickle
from matrix_class import DS_matrix
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
    corr : Float
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

    #i = 0
    matrix = DS_matrix(matrix_filename)

    for word1, word2, sim in gold_list:
        
        matrix_words = matrix.get_words()
        if not word1 in matrix_words or not word2 in matrix_words:
            continue

        #i += 1

        vec1 = matrix.get_vector(word1)
        vec2 = matrix.get_vector(word2)

        similarity = 1 - cosine(vec1, vec2)
        matrix_similarity_list.append(similarity)

        gold_similarity_list.append(float(sim))

    corr, _ = spearmanr(matrix_similarity_list, gold_similarity_list)

    #print(i)

    return corr


    
            




            
#spearman = calculate_spearman("../MEN/MEN_dataset_natural_form_full", "../bigram_matrix_10000_stopwords.pkl")
spearman = calculate_spearman("../MEN/MEN_dataset_natural_form_full", "../bigram_matrix_dict_complete.pkl")
#spearman = calculate_spearman("../MEN/MEN_dataset_natural_form_full", "../bigram_matrix_10000.pkl")
print("spearman coefficient:")
print(spearman)

