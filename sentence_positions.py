from matrix_class import DS_matrix
from scipy.spatial.distance import cosine
from scipy.cluster.vq import whiten, kmeans2
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np
import nltk
import math
import pickle
import os.path


def find_positions(sents_filename,
                   function_words,
                   matrix_filename,
                   log_filename):
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

            if s1 not in closest_sents:
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

            if s1 not in closest_words:
                closest_words[s1] = [(w, dist)]
            else:
                closest_words[s1].append((w, dist))

    with open(log_filename, "w") as log_file:
        for s1 in sents:
            log_file.write(s1)

            word_dists = closest_words[s1]

            word_dists = sorted(word_dists, key=(lambda x: x[1]))

            sent_words = nltk.word_tokenize(s1)
            sent_words = list(map(lambda w: w.lower(), sent_words))

            for i, (word, dist) in word_dists:
                if word in sent_words:
                    log_file.write(word)
                    log_file.write(" " + str(i))
                    log_file.write("\n")

            log_file.write("\n")


def find_clusters(sent_filename,
                  matrix_filename,
                  log_filename,
                  clust_dir,
                  num_clusters=2):
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
    clust_dir : str
        Directory to save the cluster to.
    num_clusters : int
        Number of clusters.
    """

    matrix = DS_matrix(matrix_filename)

    with open(sent_filename) as sent_file:
        sents = sent_file.readlines()

    sents_and_words = sents + matrix.get_words()[:5000]

    vectors = None

    for i, sent in enumerate(sents_and_words):
        vec = matrix.encode_sentence(sent)
        if vectors is None:
            vectors = csr_matrix((len(sents_and_words), vec.shape[1]))
        if len(vec.shape) > 1:
            vec = vec.reshape((vec.shape[1],))
        vectors[i] = vec

    vectors = vectors.toarray()

    vectors = whiten(vectors)

    centroids, which_cluster = kmeans2(vectors, num_clusters, minit='points')

    sent_clusters = [[] for _ in range(num_clusters)]
    vector_clusters = [[] for _ in range(num_clusters)]

    triples = zip(which_cluster, zip(vectors, sents_and_words))
    for which, (vec, sent) in triples:
        sent_clusters[which].append(sent)
        vector_clusters[which].append(vec)

    # calculate some statistics for each cluster
    dist_lists = [[] for _ in range(num_clusters)]

    vector_cluster_filename = os.path.join(clust_dir, "vector_clusters.pkl")
    with open(vector_cluster_filename, "wb") as vector_cluster_file:
        pickle.dump(vector_clusters, vector_cluster_file)
    sent_cluster_filename = os.path.join(clust_dir, "sent_clusters.pkl")
    with open(sent_cluster_filename, "wb") as sent_cluster_file:
        pickle.dump(sent_clusters, sent_cluster_file)

    with open(log_filename, "w") as log_file:
        for cluster_num, cluster in enumerate(sent_clusters):
            sents = list(filter(lambda x: x in sents, cluster))
            num_sents = len(sents)
            num_words = len(cluster) - num_sents

            log_file.write("cluster_num: " + str(cluster_num) + "\n")
            log_file.write("num_sents: " + str(num_sents) + "\n")
            log_file.write("num_words: " + str(num_words) + "\n")
            log_file.write("\n")

            log_file.write("\n\n")
            for sent in sents:
                log_file.write(sent + "\n")

            log_file.write("\n\n\n")


def cluster_stats(clust_dir, log_filename, sent_filename):
    """
    Calculate some statistics on clusters found in find_clusters.

    Parameters
    ----------
    clust_dir : str
        Directory containing the clusters.
    log_filename : str
        File to write the results to.
    sent_filename : str
        File containing the sentences of the clusters.
    """

    with open(sent_filename) as sent_file:
        sents = sent_file.readlines()

    vector_cluster_filename = os.path.join(clust_dir, "vector_clusters.pkl")
    with open(vector_cluster_filename, "rb") as vector_cluster_file:
        vector_clusters = pickle.load(vector_cluster_file)
    sent_cluster_filename = os.path.join(clust_dir, "sent_clusters.pkl")
    with open(sent_cluster_filename, "rb") as sent_cluster_file:
        sent_clusters = pickle.load(sent_cluster_file)

    dist_lists = [[] for _ in range(len(vector_clusters))]

    with open(log_filename, "w") as log_file:
        for cluster_num, cluster in enumerate(sent_clusters):
            sents = list(filter(lambda x: x in sents, cluster))
            num_sents = len(sents)
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

            log_file.write("avg dist between points: "
                           + str(np.nanmean(dist_lists[cluster_num])) + "\n")
            if len(dist_lists[cluster_num]) > 0:
                log_file.write("max dist between points: "
                               + str(max(dist_lists[cluster_num])) + "\n")
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

                        if (closest_point_dist is None
                                or closest_point_dist > dist):
                            closest_point_dist = dist
                            closest_point_dist_to_mean = cosine(vec2, mean1)

            log_file.write("closest cluster mean: "
                           + str(closest_cluster) + "\n")
            log_file.write("distance to closest cluster mean: "
                           + str(closest_cluster_dist) + "\n")

            log_file.write("distance between closest two points "
                           + "of this cluster and another one: "
                           + str(closest_point_dist) + "\n")
            log_file.write("distance of the point in the other"
                           + "cluster to the mean of this cluster: "
                           + str(closest_point_dist_to_mean) + "\n")

            log_file.write("\n\n")
            for sent in sents:
                log_file.write(sent + "\n")

            log_file.write("\n\n\n")


def find_zeros(sent_filename, matrix_filename, log_filename, num_words=None):
    """
    For each sentence and word in the matrix,
    find the number of non-zero entries in the vector.
    This is to test if the position differences are due to sparsity issues.

    Parameters
    ----------
    sent_filename : str
        Filename of the file containing the sentences to be clustered.
    matrix_filename : str
        Filename of the matrix.
    log_filename : str
        Filename of logfile.
    num_words : int
        Number of words from matrix to use.
    """

    matrix = DS_matrix(matrix_filename)

    sents_and_words = []

    with open(sent_filename) as sent_file:
        for line in sent_file:
            sents_and_words.append(line.strip())

    if num_words is None:
        sents_and_words += matrix.get_words()
    else:
        sents_and_words += matrix.get_words()[:num_words]

    pairs = []

    for w in sents_and_words:
        vec = matrix.encode_sentence(w)
        non_zero = np.count_nonzero(vec)
        pairs.append((w, non_zero))

    pairs = sorted(pairs, key=(lambda x: x[1]))

    with open(log_filename, "w") as log_file:
        for w, nz in pairs:
            line = str(nz) + " " + w + "\n"
            log_file.write(line)
