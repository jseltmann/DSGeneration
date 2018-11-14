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

            #if len(highscore_table) < 10:
            #    highscore_table.append((sim, word1, word2))
            #    highscore_table.sort(key=lambda x: x[0], reverse=True)
            #else:
            #    if highscore_table[9][0] < sim:
            #        highscore_table = highscore_table[:9]
            #        highscore_table.append((sim, word1, word2))
            #        highscore_table.sort(key=lambda x: x[0], reverse=True)
            #
            #if len(lowscore_table) < 10:
            #    lowscore_table.append((sim, word1, word2))
            #    lowscore_table.sort(key=lambda x: x[0], reverse=False)
            #else:
            #    if lowscore_table[9][0] > sim:
            #        lowscore_table = lowscore_table[:9]
            #        lowscore_table.append((sim, word1, word2))
            #        lowscore_table.sort(key=lambda x: x[0], reverse=False)
                    

    if print_results:
        bin_lengths = list(map(len, bins))
        total = sum(bin_lengths)
        bin_rel = list(map(lambda x: x/total, bin_lengths))
        print(bin_rel)
        #print()
        #print()
        #print()

        #print("Highscore:")
        #for sim, word1, word2 in highscore_table:
        #    print(sim, word1, word2)

        #print()
        #print()
        #print()

        #print("Lowscore:")
        #for sim, word1, word2 in lowscore_table:
        #    print(sim, word1, word2)

    return bins[0]#, highscore_table, lowscore_table


#bins10k = analyze_sim("../similarities/matrix_10000.pkl", "../bigram_matrix_10000.pkl_order")
#binsPca = analyze_sim("../similarities/matrix_10000_pca_300.pkl", "../bigram_matrix_10000.pkl_order")
#               
#print(len(bins10k))
#count = len(bins10k)
##for pair in bins10k:
##    if not pair in binsPca:
##        count += 1
#for pair in binsPca:
#    if pair in bins10k:
#        count -= 1
#
#print(count)
      
            
