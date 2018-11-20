import pickle
import numpy as np

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
            return 0
            #raise Exception("Word not in matrix")

        if not prev_word in self.matrix:
            return 0
            #raise Exception("Previous word not in matrix")

        vec_pos = self.vector_order[prev_word]

        return self.matrix[word][vec_pos]

    def get_words(self):
        """
        Return words contained in the matrix.
        """

        return list(self.matrix.keys())

    def generate_bigram_sentence(self, start_word=0):
        """
        Generate a sentence according to the bigram probabilities in the matrix.
        
        Parameter
        ---------
        start_word : string or 0 or 1
            First word from which to generate sentences.
            The integer 0 serves as sentence beginning symbol,
            the integer 1 as sentence end symbol.

        Return
        ------
        sentence : [String]
            Generated sentence, without beginning and end symbols.
        """


        
        sentence = []

        if start_word != 0 and start_word != 1:
            start_word = start_word.lower()
            sentence.append(start_word)

        if not start_word in self.matrix:
            raise Exception("given start_word not in matrix")

        word = start_word

        words = self.get_words()
        
        while word != 1:
            prob_list = []

            for next_word in words:
                prob = self.get_bigram_prob(next_word, word)
                prob_list.append(prob)

            index = np.random.choice(range(len(words)), p=prob_list)
            word = words[index]

            if word == 1:
                break
            
            sentence.append(word)

        return sentence

    def sentence_prob(self,sentence):
        """
        Get the probability of a sentence according to the bigram model.

        Parameter
        ---------
        sentence : [String]
            Sentence of which to get the probability.

        Returns
        -------
        prob : float
            Probability of that sentence.
        """

        prob = 1
        prev_word = 0

        for word in sentence:
            prob *= self.get_bigram_prob(prev_word, word)
            prev_word = word

        prob *= self.get_bigram_prob(prev_word,1)

        return prob
            
                                          
            
                    
    
