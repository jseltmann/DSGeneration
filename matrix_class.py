import pickle
from copy import deepcopy
import numpy as np
import scipy.sparse
import nltk

class DS_matrix:

    def __init__(self, matrix_path=None):
        if not matrix_path is None:
            with open(matrix_path, "rb") as matrix_file:
                self.matrix = pickle.load(matrix_file).tocsc()
            prefix = matrix_path[:-11]

            order_path = prefix + "_vector_index.pkl"
            with open(order_path, "rb") as order_file:
                self.vocab_order = pickle.load(order_file)

            unigram_path = prefix + "_unigram_probs.pkl"
            with open(unigram_path, "rb") as unigram_file:
                self.unigram_probs = pickle.load(unigram_file)
        else:
            self.matrix = None
            self.vocab_order = dict()
            self.unigram_probs = dict()


    def get_vector(self, word):
        """
        Return the vector that represents word.
        """
        if not word in self.vocab_order:
            raise Exception("Word not in matrix")

        pos = self.vocab_order[word]
        
        return self.matrix[pos].toarray()

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
            raise Exception("Word not in matrix")

        if not prev_word in self.vocab_order:
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
        start_word : string
            First word from which to generate sentences.

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

            pos = self.vocab_order[word]
            prob_list = self.matrix.getcol(pos).toarray().flatten()

            if sum(prob_list) == 0:
                #deal with possible cases where a word was never seen as first word in bigram using backoff
                #happens when model was trained using stopwords
                prob_list = []
                for next_word in words:
                    prob = self.get_unigram_prob(next_word)
                    prob_list.append(prob)
                
                index = np.random.choice(range(len(words)), p=prob_list)
                word = words[index]
                while word == "START$_":
                    #we don't want to have START$_ in the middle of the sentence
                    index = np.random.choice(range(len(words)), p=prob_list)
                    word = words[index]

            else:
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

        prob = self.unigram_probs[word]

        return prob


    def encode_sentence(self, sent):
        """
        Encode a sentence as sum of word vectors.
        Unknown words are ignored.

        Parameter
        ---------
        sent : String
            Sentence to be encoded.

        Return
        ------
        encoding : np.ndarray
            Vector representing the words of the sentence.
        """

        words = nltk.word_tokenize(sent)
        vectors = []

        for word in words:
            if word.lower() in self.vocab_order:
                vectors.append(self.get_vector(word.lower()))

        if len(vectors) == 0:
            encoding = np.zeros((1,self.matrix.shape[1]))
        else:
            encoding = np.sum(vectors, axis=0)

        return encoding

    def less_words_matrix(self, word_set, normalize=False):
        """
        Return a DS_matrix whose matrix contains less rows (so as to have a smaller set of words),
        but the same number of columns so that each word retains its original encoding.

        Parameters
        ----------
        word_set : [str]
            Words whose rows are to be retained. 
            Words not contained in the original matrix are ignored.
        normalize : bool
            If true, normalize the bigram probabilities.
            If false, the resulting matrix can't be used as a bigram model.

        Return
        ------
        new_matrix : DS_matrix
            New DS_matrix without the specific rows.
        """

        new_matrix = DS_matrix()

        contained_words = set()
        for word in word_set:
            if word in self.vocab_order:
                contained_words.add(word)
        word_set = contained_words

        word_set.add("START$_")
        word_set.add("END$_")

        new_matrix.matrix = scipy.sparse.lil_matrix((len(word_set), len(self.vocab_order)))

        for i, word in enumerate(word_set):
            new_matrix.vocab_order[word] = i
            new_matrix.unigram_probs[word] = self.unigram_probs[word]

            pos = self.vocab_order[word]
            new_matrix.matrix[i] = self.matrix.getrow(pos)

        new_matrix.tocsc()

        if normalize:
            #normalize probabilities
            rows, columns = new_matrix.matrix.nonzero()
            entry_dict = dict()
            for row, col in zip(rows, columns):
                if col in entry_dict.keys():
                    entry_dict[col].append(row)
                else:
                    entry_dict[col] = [row]

            for i, col in enumerate(columns):
                if i % 1000 == 0:
                    print(i)
                prob_sum = new_matrix.matrix.getcol(col).sum()

                for row in entry_dict[col]:
                    new_matrix.matrix[row, col] /= prob_sum

        return new_matrix

    def tocsr(self):
        """
        Transform self.matrix to scipy.sparse.csr_matrix.
        This is useful for row slicing, 
        for example when get_vector() is called many times.
        """

        self.matrix = self.matrix.tocsr()

    def tocsc(self):
        """
        Transform self.matrix to scipy.sparse.csc_matrix.
        This is useful for column slicing,
        for example when generate_bigram_sentence is called many times.
        """

        self.matrix = self.matrix.tocsc()
            
        



