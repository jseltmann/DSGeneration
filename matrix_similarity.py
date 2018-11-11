from matrix_class import DS_matrix
from scipy.spatial.distance import cosine
import numpy as np
import pickle

def similarity(vector1, vector2):
    """
    Calculate cosine similarity of two vectors.
    """

    similarity = 1 - cosine(vector1, vector2)

    return similarity


def similarity_matrix(matrix_path, similarity_path, dict_path=None):
    """
    For each pair of words in the matrix,
    calculate the similarity between the
    corresponding vectors and save it in
    a new matrix.

    Parameter
    ---------
    matrix_path : String
        path to file containing the matrix
    similarity_path : String
        path to save the similarity matrix to
    dict_path : String
        additional path to save the similarity matrix as dict
    """

    matrix = DS_matrix(matrix_path)

    words = matrix.get_words()
    word_num = len(words)

    i = 0
    j = 0
    similarity_matrix = np.empty((word_num, word_num), float)

    if dict_path:
        similarity_dict = dict()
    
    
    for i, word1 in enumerate(words):
        if dict_path:
            similarity_dict[word1] = dict()
        for j, word2 in enumerate(words):
            vec1 = matrix.get_vector(word1)
            vec2 = matrix.get_vector(word2)

            sim = similarity(vec1, vec2)
            similarity_matrix[i][j] = sim

            if dict_path:
                similarity_dict[word1][word2] = sim

        if i % 500 == 0:
            print(i)
       



    with open(similarity_path, "wb") as sim_file:
        pickle.dump(similarity_matrix, sim_file)

    if dict_path:
        with open(dict_path, "wb") as dict_file:
            pickle.dump(similarity_dict, dict_file)
        

similarity_matrix("../bigram_matrix_10000.pkl", "../similarities/matrix_10000.pkl", "../similarities/dict_10000.pkl")
similarity_matrix("../bigram_matrix_10000_pca_300.pkl", "../similarities/matrix_10000_pca_300.pkl", "../similarities/dict_10000_pca_300.pkl")

            
