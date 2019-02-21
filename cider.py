import json
import pickle
from matrix_class import DS_matrix
import collections as col
import nltk
import numpy as np
import math

def calculate_ciderD(decoded_filename, ref_filename, log_filename, max_n=4):
    """
    Calculate ciderD scores for decoded sentences.

    Parameters
    ----------
    decoded_filename : str
        File containing the decoded sentences.
    ref_filename : str
        File containing reference sentences for each decoded sentence.
    log_filename : str
        File to write the results to.
    max_n : int
        Maximum size of ngrams to use for ciderD calculation.
    """

    with open(decoded_filename, "rb") as decoded_file:
        decoded_dict = pickle.load(decoded_file)
    with open(ref_filename, "rb") as ref_file:
        ref_dict = pickle.load(ref_file)

    scores = []

    ngram_counts = get_ngram_counts(ref_dict, max_n)
    
    #i = 0
    for image_id, dec_sent in decoded_dict.items():
        #if i % 50 == 0:
        #    print(i)
        #if i < 10:
        #    print(i)
        #i += 1
        score = 0
        
        for n in list(range(max_n+1))[1:]:
            score += 1/max_n * calculate_ciderD_n(image_id, dec_sent, ref_dict, ngram_counts, n)

        scores.append(score)

        with open(log_filename, "a") as log_file:
            line = str(image_id) + " | " + str(dec_sent) + " | " + str(score) + "\n"
            log_file.write(line)

    avg = np.mean(scores)
    stddev = np.std(scores)

    with open(log_filename, "a") as log_file:
        log_file.write("\n\n")
        log_file.write("average ciderD score: " + str(avg) + "\n")
        log_file.write("standard deviation: " + str(stddev) + "\n")

def calculate_ciderD_n(cand_id, cand_sent, ref_dict, ngram_counts, n, sigma=6):
    """
    Calculate ciderD score for ngrams of length n.

    Parameters
    ----------
    cand_id : str
        Image id of candidate sentence.
    cand_sent : [str]
        Candidate sentence.
    ref_dict : dict([[str]])
        Dictionary containing the reference sentences for each image id.
    n : int
        Length of ngrams considered in this function call.
    sigma : float
        Parameter to control length difference penalty.

    Return
    ------
    ciderD_n : float
        ciderD score for ngrams of length n of this sentence.
    """

    score = 0

    cand_vec = get_vector(cand_sent, ref_dict, ngram_counts, n)

    #i = 0
    for ref_sent in ref_dict[cand_id]:
        #print(i)
        #i += 1
        ref_vec = get_vector(ref_sent, ref_dict, ngram_counts, n)
        clipped = clip_vector(cand_vec, ref_vec)

        penalty = math.exp( -(len(cand_sent) - len(ref_sent))**2 / (2 * sigma**2))

        norm_clipped = norm(clipped)
        norm_ref = norm(ref_vec)

        if norm_clipped * norm_ref != 0:
            score += penalty * dot(clipped, ref_vec) / (norm(clipped) * norm(ref_vec))

    score *= 10/len(ref_dict[cand_id])

    return score



def norm(vec):
    """
    Get norm of vector.

    Parameters
    ----------
    vec : dict(float)
        Vector of which to get the norm.

    Return
    ------
    norm : float
        Norm of the vector.
    """

    norm_squared = 0

    for key in vec:
        norm_squared += vec[key] * vec[key]

    norm = norm_squared ** (1/2)

    return norm


def dot(vec1, vec2):
    """
    Calculate dot product of two vectors.
    
    Parameters
    ----------
    vec1 : dict(float)
        First vector.
    vec2 : dict(float)
        Second vector.

    Return
    ------
    dot_prod : float
        Dot product of vec1 and vec2.
    """

    keys = set()
    for key in vec1.keys():
        keys.add(key)
    for key in vec2.keys():
        keys.add(key)

    dot_prod = 0
        
    for key in keys:
        if not key in vec1 or not key in vec2:
            continue
        dot_prod += vec1[key] * vec2[key]

    return dot_prod


def clip_vector(vec1, vec2):
    """
    Clip values in vec1 to the ones in vec2.

    Parameters
    ----------
    vec1 : dict(float)
        Vector, whose values are to be clipped.
    vec2 : dict(float)
        Vector given the maximum values for clipping.

    Return
    ------
    clipped : dict(float)
        Clipped vector.
    """

    clipped = col.defaultdict()
    
    for ngram in vec1:
        if not ngram in vec2:
            clipped[ngram] = 0
        elif vec1[ngram] > vec2[ngram]:
            clipped[ngram] = vec2[ngram]
        else:
            clipped[ngram] = vec1[ngram]

    return clipped


def get_vector(sent, ref_dict, ngram_counts, n):
    """
    Get vector of ngram tfidf values which represents this sentence.

    Parameters
    ----------
    sent : [str]
        Sentence for which to get tfidf representation.
    ref_dict : dict([[str]])
        Reference sentences for each image id.
    n : int
        Length of ngrams to consider

    Return
    ------
    vector : dict(float)
        Vector of ngram tfidf values. The dict keys are the ngrams.
    """
    vector = col.defaultdict(float)

    ngrams = [tuple(sent[i:i+n]) for i in range(len(sent) - n + 1)]

    for ngram in ngrams:
        vector[ngram] = tfidf(ngram, ngrams, ref_dict, ngram_counts, n)

    return vector


def tfidf(ngram, ngrams, ref_dict, ngram_counts, n):
    """
    Compute tfidf for ngram.

    Parameters
    ----------
    ngram : tuple(str)
        Ngram for which to compute tfidf.
    ngrams : [tuple(str)]
        All ngrams of the sentence from which ngram comes.
    ref_dict : dict([[str]])
        Reference sentences for each image id.
    n : int
        Length of ngrams in question.

    Return
    ------
    tfidf : float
        tfidf of ngram.
    """
    sent_counts = col.defaultdict()

    for sent_ngram in ngrams:
        sent_counts[sent_ngram] = ngrams.count(sent_ngram)
    

    tf = sent_counts[ngram] / (sum(sent_counts.values()))

    df = ngram_counts[ngram]
    if df == 0: #for ngrams that never occured in the reference sentences
        df = len(ref_dict.keys())
    
    idf = np.log(len(ref_dict.keys()) / df)

    tfidf = tf * idf

    return tfidf


def get_ngram_counts(ref_dict, max_n):
    """
    For each ngram, add the appearences of that ngram in the references for each image.
    If it doesn't appear in the references for an image, add 1.
    This is for the denominator of the idf part of the tfidf calculation.

    Parameters
    ----------
    ref_dict : dict([[str]])
        Reference sentences for each image id.
    max_n : int
        Maximum length of ngrams.

    Return
    ------
    ngram_counts : dict(int)
        Dictionary giving the count for each ngram in the reference sentences.
    """

    ngrams = set()

    for sent_list in ref_dict.values():
        for sent in sent_list:
            sent_ngrams = []
            for n in list(range(max_n+1))[1:]:
                sent_ngrams += [tuple(sent[i:i+n])  for i in range(len(sent) - n + 1)]
            for sent_ngram in sent_ngrams:
                ngrams.add(tuple(sent_ngram))

    ngram_counts = col.defaultdict(int)
    
    for image_id in ref_dict.keys():
        image_counts = col.defaultdict(int)
        for ref_sent in ref_dict[image_id]:
            ref_ngrams = []
            for n in list(range(max_n+1))[1:]:
                ref_ngrams += [tuple(ref_sent[i:i+n]) for i in range(len(ref_sent) - n + 1)]
            for ref_ngram in ref_ngrams:
                image_counts[ref_ngram] += ref_ngrams.count(ref_ngram)
        for ngram in ngrams:
            if image_counts[ngram] == 0:
                ngram_counts[ngram] += 1
            else:
                ngram_counts[ngram] += image_counts[ngram]

    return ngram_counts
        
        
    
            
        



    




def decode_sentences(data_filename, decoded_filename, ref_filename, matrix_filename):
    """
    Read sentences from the PASCAL50S dataset. For each image, encode and decode one and save the others to a file.
    
    Parameters
    ----------
    data_filename : str
        File containing the dataset.
    decoded_filename : str
        File to save the decoded sentences to as a dict of image_ids and sentences.
    ref_filename : str
        File to save the remaining sentences to as a dict of image_ids and lists of sentences.
    matrix_filename : str
        File containing the DS matrix.
    """

    with open(data_filename) as f:
        data = json.loads(f.read())


    orig_dict = dict()
    decoded_dict = dict()
    ref_dict = dict()

    i = 0
    for entry in data:
        image_id = entry['image_id']
        sent = entry['caption']
        if not image_id in orig_dict:
            orig_dict[image_id] = sent
        elif not image_id in ref_dict:
            ref_dict[image_id] = [nltk.word_tokenize(sent)]
        else:
            ref_dict[image_id].append(nltk.word_tokenize(sent))
            

    matrix = DS_matrix(matrix_filename)

    for image_id, sent in orig_dict.items():
        decoded = matrix.reconstruct_sent(sent)
        decoded_dict[image_id] = nltk.word_tokenize(decoded)

    with open(decoded_filename, "wb") as decoded_file:
        pickle.dump(decoded_dict, decoded_file)

    with open(ref_filename, "wb") as ref_file:
        pickle.dump(ref_dict, ref_file)


#decode_sentences("../cider/data/pascal50S.json", "../test_decoded.pkl", "../test_ref.pkl", "../matrix_1k/_matrix.pkl")
#decode_sentences("../cider/data/pascal50S.json", "../debug_decoded.pkl", "../debug_ref.pkl", "../matrix_1k/_matrix.pkl")
                 
#calculate_ciderD("../test_decoded.pkl", "../test_ref.pkl", "../test.log", max_n=4)
