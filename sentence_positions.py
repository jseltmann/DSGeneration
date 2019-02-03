from matrix_class import DS_matrix
from scipy.spatial.distance import cosine
from scipy.cluster.vq import whiten, kmeans2
from collections import defaultdict
import numpy as np
import nltk
import math

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

            #if not s1 in closest_words:
            #    closest_words[s1] = (w, dist)
            #else:
            #    if closest_words[s1][1] > dist:
            #        closest_words[s1] = (w, dist)
            if not s1 in closest_non_func_words:
                closest_non_func_words[s1] = [(w, dist)]
            elif len(closest_non_func_words[s1]) > 9:
                furthest = max(closest_non_func_words[s1], key=(lambda x: x[1]))
                if furthest[1] > dist:
                    closest_non_func_words[s1].remove(furthest)
                    closest_non_func_words[s1].append((w, dist))
            else:
                closest_non_func_words[s1].append((w, dist))

        if w not in function_words:
            for s1 in sents:
                vec1 = sent_vectors[s1]

                dist = cosine(vec1, vec2)

                if not s1 in closest_non_func_words:
                    closest_non_func_words[s1] = [(w, dist)]
                elif len(closest_non_func_words[s1]) > 9:
                    furthest = max(closest_non_func_words[s1], key=(lambda x: x[1]))
                    if furthest[1] > dist:
                        closest_non_func_words[s1].remove(furthest)
                        closest_non_func_words[s1].append((w, dist))
                else:
                    closest_non_func_words[s1].append((w, dist))
                    #if closest_non_func_words[s1][1] > dist:
                    #    closest_non_func_words[s1] = (w, dist)
            

    with open(log_filename, "w") as log_file:
        for s1 in sents:
            log_file.write(s1)
            
            #closest_sent, dist = closest_sents[s1]
            #log_file.write(closest_sent)
            #log_file.write(str(dist) + "\n")

            #closest_word, dist = closest_words[s1]
            #log_file.write(closest_word + "\n")
            #log_file.write(str(dist) + "\n")

            #closest_word, dist = closest_non_func_words[s1]
            #log_file.write(closest_word + "\n")
            #log_file.write(str(dist) + "\n")
            log_file.write(str(closest_words[s1]))
            log_file.write("\n")
            log_file.write(str(closest_non_func_words[s1]))
            log_file.write("\n")

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
        #values = list(map(lambda x:x[1], closest_non_func_words.values()))
        #dist_avg = np.mean(values)
        #dist_var = np.var(values)

        #log_file.write("Average distance of closest non-function word to sentence:\n")
        #log_file.write("avg: " + str(dist_avg) + "\n")
        #log_file.write("var: " + str(dist_var) + "\n")

        #log_file.write("\n\n")


        #contained_count = 0
        #contained_non_func = 0

        #for sent in sents:
        #    words = map(lambda x: x.lower(), nltk.word_tokenize(sent))
        #    if closest_words[sent][0] in words:
        #        contained_count += 1
        #    if closest_non_func_words[sent][0] in words:
        #        contained_non_func += 1
        #            
        #log_file.write("Fraction of sentences when closest word is contained in sentence:\n")
        #log_file.write(str(contained_count/len(closest_words)) + "\n")

        #log_file.write("Fraction of sentences when closest non-function word is contained in sentence:\n")
        #log_file.write(str(contained_non_func/len(closest_non_func_words)) + "\n")


def find_clusters(sent_filename, matrix_filename, log_filename, num_clusters=2):
    """
    Use k-means clustering to find clusters in space among 
    given sentences and the words in a DS_matrix.

    Parameters
    ----------
    sent_filename : str
        Filename of the file containing the sentences to be clustered.
    matrix_filename : str
        Filename of the matrix.
    log_filename : str
        Filename of logfile.
    num_clusters : int
        Number of clusters.
    """

    matrix = DS_matrix(matrix_filename)

    with open(sent_filename) as sent_file:
        sents = sent_file.readlines()

    sents_and_words = sents + matrix.get_words()

    vectors = []

    for sent in sents_and_words:
        vec = matrix.encode_sentence(sent).reshape((1002,))
        vectors.append(vec)

    vectors = np.array(vectors)

    vectors = whiten(vectors)

    centroids, which_cluster = kmeans2(vectors, num_clusters, minit='points')


    sent_clusters = [[] for _ in range(num_clusters)]
    vector_clusters = [[] for _ in range(num_clusters)]

    for which, (vec, sent) in zip(which_cluster, zip(vectors, sents_and_words)):
        sent_clusters[which].append(sent)
        vector_clusters[which].append(vec)
    
    #calculate some statistics for each cluster
    dist_lists = [[] for _ in range(num_clusters)]

    with open(log_filename, "w") as log_file:
        for cluster_num, cluster in enumerate(sent_clusters):
            num_sents = len(list(filter(lambda x: x in sents, cluster)))
            num_words = len(cluster) - num_sents

            log_file.write("cluster_num: " + str(cluster_num) + "\n")
            log_file.write("num_sents: " + str(num_sents) + "\n")
            log_file.write("num_words: " + str(num_words) + "\n")
            log_file.write("\n")

            vectors = vector_clusters[cluster_num]

            for i, vec1 in enumerate(vectors):
                for j, vec2 in enumerate(vectors):
                    if i == j:
                        continue
                    dist = cosine(vec1, vec2)
                    dist_lists[cluster_num].append(dist)

            log_file.write("avg dist between points: " + str(np.nanmean(dist_lists[cluster_num])) + "\n")
            if len(dist_lists[cluster_num]) > 0:
                log_file.write("max dist between points: " + str(max(dist_lists[cluster_num])) + "\n")
            else:
                log_file.write("max dist between points:")

            mean1 = np.mean(vectors, axis=0)

            closest_cluster = None
            closest_cluster_dist = None

            closest_point_dist = None

            for cluster_num2, cluster2 in enumerate(vector_clusters):
                if cluster_num2 == cluster_num:
                    continue

                mean2 = np.mean(cluster2, axis=0)
                dist = cosine(mean1, mean2)

                if closest_cluster_dist is None or closest_cluster_dist > dist:
                    closest_cluster = cluster_num2
                    closest_cluster_dist = dist

                for vec1 in vectors:
                    for vec2 in cluster2:
                        dist = cosine(vec1, vec2)

                        if closest_point_dist is None or closest_point_dist > dist:
                            closest_point_dist = dist
                            closest_point_dist_to_mean = cosine(vec2, mean1)

            log_file.write("closest cluster mean: " + str(closest_cluster) + "\n")
            log_file.write("distance to closest cluster mean: " + str(closest_cluster_dist) + "\n")

            log_file.write("distance between closest two points of this cluster and another one: " + str(closest_point_dist) + "\n")
            log_file.write("distance of the point in the other cluster to the mean of this cluster: " + str(closest_point_dist_to_mean) + "\n")

            log_file.write("\n\n\n")

                
    



                    

stopwords = [
    "", "(", ")", "a", "about", "an", "and", "are", "around", "as", "at",
    "away", "be", "become", "became", "been", "being", "by", "did", "do",
    "does", "during", "each", "for", "from", "get", "have", "has", "had", "he",
    "her", "his", "how", "i", "if", "in", "is", "it", "its", "made", "make",
    "many", "most", "not", "of", "on", "or", "s", "she", "some", "that", "the",
    "their", "there", "this", "these", "those", "to", "under", "was", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your"
] + [".", ","]

#find_positions("../combined_sents.txt", stopwords, "../matrix_50k/_matrix.pkl", "../sent_pos_non_func_10.log") 
