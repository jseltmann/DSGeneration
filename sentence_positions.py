from matrix_class import DS_matrix
from scipy.spatial.distance import cosine

def find_positions(sents_filename, matrix_filename, log_filename):
    """
    For each sentence in the sents_file, find the sentence in sents_file 
    and the word in the matrix, whose vectors are closest to the vector 
    of this sentence, measured by cosine distance.

    Parameters
    ----------
    sents_filename : str
        Filename of the file containing the sentences.
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

    for s2 in sents:
        vec2 = sent_vectors[s2]
        for s1 in sents:
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
                closest_words[s1] = [(s2, dist)]
            else:
                if len(closest_words[s1]) >= 10:
                    closest_words.remove(max(closest_words, key=(lambda t:t[1])))
                closest_words[s1].append((s2, dist))

    with open(log_filename, "w") as log_file:
        for s1 in sents:
            log_file.write(s1)
            
            closest_sent, dist = closest_sents[s1]
            log_file.write(closest_sent)
            log_file.write(str(dist) + "\n")

            closest_word, dist = closest_words[s1]
            log_file.write(closest_word + "\n")
            log_file.write(str(dist) + "\n")

            log_file.write("\n\n")

                    
