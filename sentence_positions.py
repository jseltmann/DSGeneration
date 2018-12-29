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

    sent_vectors = dict()
    for sent in sents:
        sent_vectors[sent] = matrix.encode_sentence(sent)


    for i, s1 in enumerate(sents):
        print(i)
        min_dist = float("inf")
        closest_sent = None
        vec1 = sent_vectors[s1]

        for s2 in sents:
            if s1 == s2:
                continue
            vec2 = sent_vectors[s2]

            dist = cosine(vec1, vec2)

            if dist < min_dist:
                min_dist = dist
                closest_sent = s2

        with open(log_filename, "a") as log_file:
            log_file.write(s1)
            log_file.write(closest_sent)
            log_file.write(str(min_dist) + "\n")

        min_dist = float("inf")
        closest_word = None
        for j, w in enumerate(matrix.get_words()):
            #if j % 500 == 0:
            #    print(j)
            if w == s1.strip():
                continue
            vec2 = matrix.get_vector(w)

            dist = cosine(vec1, vec2)

            if dist < min_dist:
                min_dist = dist
                closest_sent = w

        with open(log_filename, "a") as log_file:
            log_file.write(w + "\n")
            log_file.write(str(min_dist))
            log_file.write("\n\n\n")
