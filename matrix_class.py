import pickle
from copy import deepcopy
import numpy as np

class DS_matrix:

    def __init__(self, matrix_path, order_path):
        with open(matrix_path, "rb") as matrix_file:
            self.matrix = pickle.load(matrix_file)
        with open(order_path, "rb") as order_file:
            self.vocab_order = pickle.load(order_file)

    def get_vector(self, word):
        """
        Return the vector that represents word.
        """
        if not word in self.vocab_order:
            raise Exception("Word not in matrix")

        pos = self.vocab_order[word]
        
        return self.matrix[pos]

    def contains(self, word):
        """
        Returns true if word is in matrix. False otherwise.
        """
        return word in self.vocab_order
    
    def get_bigram_prob(self, word, prev_word):
        """
        Return the probability p(word|prev_word).
        """

        if not word in self.vocab_order:
            #return 0
            raise Exception("Word not in matrix")

        if not prev_word in self.vocab_order:
            #return 0
            raise Exception("Previous word not in matrix")

        prev_pos = self.vocab_order[prev_word]
        pos = self.vocab_order[word]

        return self.matrix[pos, prev_pos]

    def get_words(self):
        """
        Return words contained in the matrix.
        """

        return list(self.vocab_order.keys())

    def generate_bigram_sentence(self, start_word="START$_"):
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

        if start_word != "START$_" and start_word != "END$_":
            start_word = start_word.lower()

        if not start_word in self.vocab_order:
            raise Exception("given start_word not in matrix")

        word = start_word

        words = self.get_words()
        
        while word != "END$_":


            prob_list = []

            for next_word in words:
                prob = self.get_bigram_prob(next_word, word)
                prob_list.append(prob)

            index = np.random.choice(range(len(words)), p=prob_list)
            word = words[index]

            if word == "END$_":
                break
            
            sentence.append(word)
            

        return sentence

    def get_sentence_prob(self, sentence):
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
        prev_word = "START$_"

        for word in sentence:
            if not word in self.vocab_order:
                continue
            prob *= self.get_bigram_prob(word, prev_word)
            prev_word = word

        prob *= self.get_bigram_prob("END$_", prev_word)

        return prob


    def get_unigram_prob(self, word):
        """
        Get the probability of a specific word ocuuring.

        Parameter
        ---------
        word : string
            Word of which to get the probability.

        Return
        ------
        prob : float
            Probability of word occuring.
        """

        pos = self.vocab_order[word]
        
        prob = self.matrix[pos].sum()

        return prob




        



