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


    
            

#pca_matrix("../bigram_matrix_10k.pkl", "../bigram_matrix_10k_pca_100.pkl", n_components=100)
#pca_matrix("../bigram_matrix_10k.pkl", "../bigram_matrix_10k_pca_200.pkl", n_components=200)
#pca_matrix("../bigram_matrix_10k.pkl", "../bigram_matrix_10k_pca_500.pkl", n_components=500)
#pca_matrix("../bigram_matrix_10k.pkl", "../bigram_matrix_10k_pca_1000.pkl", n_components=1000)

#pca_matrix("../bigram_matrix_10k.pkl", "../bigram_matrix_10k_pca_1500.pkl", n_components=1500)
#pca_matrix("../bigram_matrix_10k.pkl", "../bigram_matrix_10k_pca_2000.pkl", n_components=2000)




spearman = calculate_spearman("../MEN/MEN_dataset_natural_form_full", "../bigram_matrix_10k_new_stopwords.pkl")
print("new stopwords")
print(spearman)

spearman = calculate_spearman("../MEN/MEN_dataset_natural_form_full", "../bigram_matrix_10k_stopwords.pkl")
print("old stopwords")
print(spearman)
