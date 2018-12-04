from matrix_class import DS_matrix
from scipy.spatial.distance import cosine
import numpy as np
import pickle

def similarity_matrix(matrix_path, similarity_path, similarity=(lambda x: 1- cosine(x[0],x[1]))):
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

    
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            vec1 = matrix.get_vector(word1)
            vec2 = matrix.get_vector(word2)

            sim = similarity(vec1, vec2)
            similarity_matrix[i][j] = sim

            if dict_path:
                similarity_dict[word1][word2] = sim

        if i % 1000 == 0:
            print(i)
       



    with open(similarity_path, "wb") as sim_file:
        pickle.dump(similarity_matrix, sim_file)

    if dict_path:
        with open(dict_path, "wb") as dict_file:
            pickle.dump(similarity_dict, dict_file)
        

def analyze_sim(similarity_path, order_path, print_results=False):
    with open(similarity_path, "rb") as similarity_file:
        sim_matrix = pickle.load(similarity_file)

    with open(order_path, "rb") as order_file:
        word_order = pickle.load(order_file)

    bins = [[],[],[],[],[],[],[],[],[],[]]
    highscore_table = []
    lowscore_table = []

    for i, word1 in enumerate(word_order):
        for j, word2 in enumerate(word_order):
            if j >= i:
                break
            sim = sim_matrix[i][j]
            if sim > 0.8:
                bins[0].append((word1,word2))
            elif sim > 0.6:
                bins[1].append((word1,word2))
            elif sim > 0.4:
                bins[2].append((word1,word2))
            elif sim > 0.2:
                bins[3].append((word1,word2))
            elif sim > 0:
                bins[4].append((word1,word2))
            elif sim > -0.2:
                bins[5].append((word1,word2))
            elif sim > -0.4:
                bins[6].append((word1,word2))
            elif sim > -0.6:
                bins[7].append((word1,word2))
            elif sim > -0.8:
                bins[8].append((word1,word2))
            else:
                bins[9].append((word1,word2))

                    

    bin_lengths = list(map(len, bins))

    return bin_lengths
      
            
