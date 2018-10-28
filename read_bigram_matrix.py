import re
import numpy
import nltk


def read_bigram_matrix(corpus_filename, freqdist_filename, word_num=None):
    matrix = dict()
    vector_order = dict()

    #if word_num is None:
    #    word_num = 0
    #    with open(freqdist_filename) as freqdist_file:
    #        for line in freqdist_file:
    #            word_num += 1
        
    
    with open(freqdist_filename) as freqdist_file:
        i = 0
        reg = r"[0-9]+\s(.+)"

        for line in freqdist_file:
            if word_num and i == word_num:
                break
            m = re.match(reg, line)
            if not m:
                print(line)
                continue
            word = m.group(1)
            matrix[word] = None#np.zeros(word_num)
            vector_order[word] = i
            i += 1

        if word_num is None:
            word_num = i

    for word in matrix:
        matrix[word] = np.zeros(word_num)


    with open(corpus_filename) as corpus_file:
        for line in corpus_file:
            words = line.split()
            prev_word = None

            for word in words:
                if not word in matrix:
                    continue
                if prev_word is None:
                    prev_word = word
                    continue

                vec_pos = vector_order[prev_word]
                matrix[word][prev_word] += 1

                prev_word = word

    for word in matrix:
        #matrix[word] = normalize(matrix[word])
        total = sum(matrix[word])
        matrix[word] = matrix[word] / total


            

        

        

        
