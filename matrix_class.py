import pickle

class DS_matrix:

    def __init__(self, matrix_path, order_path=None):
        with open(matrix_path, "rb") as matrix_file:
            self.matrix = pickle.load(matrix_file)
        if not order_path is None:
            with open(order_path, "rb") as order_file:
                self.vector_order = pickle.load(order_file)
        else:
            self.vector_order = None

    def get_vector(self, word):
        """
        Return the vector that represents word.
        """
        if not word in self.matrix:
            raise Exception("Word not in matrix")

        return self.matrix[word]

    def get_bigram_prob(self, word, prev_word):
        """
        Return the probability p(word|prev_word).
        """

        if self.vector_order is None:
            raise Exception("No vector_order is given.")

        if not word in self.matrix:
            raise Exception("Word not in matrix")

        if not prev_word in self.matrix:
            raise Exception("Previous word not in matrix")

        vec_pos = self.vector_order[prev_word]

        return self.matrix[word][vec_pos]

    def get_words(self):
        """
        Return words contained in the matrix."
        """

        return self.matrix.keys()
    
