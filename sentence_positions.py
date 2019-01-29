from matrix_class import DS_matrix
from scipy.spatial.distance import cosine
from collections import defaultdict
import numpy as np
import nltk

def find_positions(sents_filename, function_words, matrix_filename, log_filename):
    """
    For each sentence in the sents_file, find the sentence in sents_file 
    and the word in the matrix, whose vectors are closest to the vector 
    of this sentence, measured by cosine distance.

    Parameters
    ----------
    sents_filename : str
        Filename of the file containing the sentences.
    function_words : [str]
        Function words to be excluded for statistic without function words.
    matrix_filename : str
        Filename of the file containing the matrix.
    log_filename : str
        Filename to which to write the sentences and 
        the most similar other sentences and words.
    """

    sents = open(sents_filename).readlines()

    matrix = DS_matrix(matrix_filename)
    matrix.tocsr()

    sent_vectors = dict()
    for sent in sents:
        sent_vectors[sent] = matrix.encode_sentence(sent)

    closest_sents = dict()
    closest_words = dict()
    closest_non_func_words = dict()


    for s2 in sents:
        vec2 = sent_vectors[s2]
        for s1 in sents:
            if s1 == s2:
                continue
            vec1 = sent_vectors[s1]

            dist = cosine(vec1, vec2)

            if not s1 in closest_sents:
                closest_sents[s1] = (s2, dist)
            else:
                if dist < closest_sents[s1][1]:
                    closest_sents[s1] = (s2, dist)

    print("found closest sentences...")
    
    for i, w in enumerate(matrix.get_words()):
        if i % 5000 == 0:
            print(i)
        vec2 = matrix.get_vector(w)
        for s1 in sents:
            vec1 = sent_vectors[s1]

            dist = cosine(vec1, vec2)

            if not s1 in closest_words:
                closest_words[s1] = (w, dist)
            else:
                if closest_words[s1][1] > dist:
                    closest_words[s1] = (w, dist)

        if w not in function_words:
            for s1 in sents:
                vec1 = sent_vectors[s1]

                dist = cosine(vec1, vec2)

                if not s1 in closest_non_func_words:
                    closest_non_func_words[s1] = (w, dist)
                else:
                    if closest_non_func_words[s1][1] > dist:
                        closest_non_func_words[s1] = (w, dist)
            

    with open(log_filename, "w") as log_file:
        for s1 in sents:
            log_file.write(s1)
            
            closest_sent, dist = closest_sents[s1]
            log_file.write(closest_sent)
            log_file.write(str(dist) + "\n")

            closest_word, dist = closest_words[s1]
            log_file.write(closest_word + "\n")
            log_file.write(str(dist) + "\n")

            closest_word, dist = closest_non_func_words[s1]
            log_file.write(closest_word + "\n")
            log_file.write(str(dist) + "\n")

            log_file.write("\n\n")
        log_file.write("\n\n\n")

        #get average distance to closest sentences
        #print(type(list(closest_sents.values())[0]))
        #print(list(closest_sents.items())[0])
        #dist_avg = np.mean(list(closest_sents.values()))
        #dist_var = np.var(list(closest_sents.values()))
        values = list(map(lambda x:x[1], closest_sents.values()))
        dist_avg = np.mean(values)
        dist_var = np.var(values)

        log_file.write("Average distance of closest other sentence to sentence:\n")
        log_file.write("avg: " + str(dist_avg) + "\n")
        log_file.write("var: " + str(dist_var) + "\n")

        log_file.write("\n\n")

        #get average distance to closest word
        #dist_avg = np.mean(list(closest_words.values()))
        #dist_var = np.var(list(closest_words.values()))
        values = list(map(lambda x:x[1], closest_words.values()))
        dist_avg = np.mean(values)
        dist_var = np.var(values)

        log_file.write("Average distance of closest word to sentence:\n")
        log_file.write("avg: " + str(dist_avg) + "\n")
        log_file.write("var: " + str(dist_var) + "\n")

        log_file.write("\n\n")

        #get average distance to closest non-function word
        #dist_avg = np.mean(list(closest_non_func_words.values()))
        #dist_var = np.var(list(closest_non_func_words.values()))
        values = list(map(lambda x:x[1], closest_non_func_words.values()))
        dist_avg = np.mean(values)
        dist_var = np.var(values)

        log_file.write("Average distance of closest non-function word to sentence:\n")
        log_file.write("avg: " + str(dist_avg) + "\n")
        log_file.write("var: " + str(dist_var) + "\n")

        log_file.write("\n\n")


        contained_count = 0
        contained_non_func = 0

        for sent in sents:
            words = map(lambda x: x.lower(), nltk.word_tokenize(sent))
            if closest_words[sent][0] in words:
                contained_count += 1
            if closest_non_func_words[sent][0] in words:
                contained_non_func += 1
                    
        log_file.write("Fraction of sentences when closest word is contained in sentence:\n")
        log_file.write(str(contained_count/len(closest_words)) + "\n")

        log_file.write("Fraction of sentences when closest non-function word is contained in sentence:\n")
        log_file.write(str(contained_non_func/len(closest_non_func_words)) + "\n")




                    

stopwords = [
    "", "(", ")", "a", "about", "an", "and", "are", "around", "as", "at",
    "away", "be", "become", "became", "been", "being", "by", "did", "do",
    "does", "during", "each", "for", "from", "get", "have", "has", "had", "he",
    "her", "his", "how", "i", "if", "in", "is", "it", "its", "made", "make",
    "many", "most", "not", "of", "on", "or", "s", "she", "some", "that", "the",
    "their", "there", "this", "these", "those", "to", "under", "was", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your"
]

find_positions("../combined_sents.txt", stopwords, "../matrix_1k/_matrix.pkl", "../test.log") 
