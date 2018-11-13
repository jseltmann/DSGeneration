import pickle
import numpy as np

class DS_matrix:

    def __init__(self, matrix_path, order_path=None):
        with open(matrix_path, "rb") as matrix_file:
            self.matrix_dict = pickle.load(matrix_file)
        #if not order_path is None:
        #    with open(order_path, "rb") as order_file:
        #        self.vector_order = pickle.load(order_file)
        #else:
        #    self.vector_order = None

    def get_vector(self, word):
        """
        Return the vector that represents word.
        """
        if not word in self.matrix_dict:
            raise Exception("Word not in matrix")

        vector = np.zeros(len(self.matrix_dict))

        for i, prev_word in enumerate(self.matrix_dict):
            if prev_word in self.matrix_dict[word]:
                vector[i] = self.matrix_dict[word][prev_word]

        return vector

    def get_bigram_prob(self, word, prev_word):
        """
        Return the probability p(word|prev_word).
        """

        if not word in self.matrix_dict:
            raise Exception("Word not in matrix")

        if prev_word in self.matrix_dict[word]:
            return self.matrix_dict[word][prev_word]
        else:
            return 0
        
    def get_words(self):
        """
        Return words contained in the matrix."
        """

        return self.matrix_dict.keys()
    
