import json
import pickle
from matrix_class import DS_matrix
import collections as col
import nltk
import numpy as np

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

    #ngram_frequencies = get_ngram_frequencies(ref_dict, max_n)

    for image_id, dec_sent in decoded_dict.items():
        score = 0
        
        for n in ngram_frequencies.keys():
            score += 1/max_n * calculate_ciderD_n(image_id, dec_sent, ref_dict, n)

def calculate_ciderD_n(cand_id, cand_sent, ref_dict, n):
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

    Return
    ------
    ciderD_n : float
        ciderD score for ngrams of length n of this sentence.
    """

    score = 0

    cand_vec = get_vector(cand_sent, ref_dict, n)

    for ref_sent in ref_dict[cand_id]:
        ref_vec = get_vector(ref_sent, ref_dict, n)
        clipped = clip_vector(cand_vec, ref_vec)


def clip_vector(####tbc


def get_vector(sent, ref_dict, n):
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
    """
    #vector and tfidf only care about ngrams of length n
    vector = col.defaultdict(float)

    ngrams = [sent[i:i+n] for i in range(len(sent) - n + 1)]

    for ngram in ngrams:
        vector[ngram] = tfidf(ngram, ngrams, ref_dict, n)

    return vector


def tfidf(ngram, ngrams, ref_dict, n):
    """
    Compute tfidf for ngram.

    Parameters
    ----------
    ngram : [str]
        Ngram for which to compute tfidf.
    ngrams : [[str]]
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

    sent_counts = col.Counter(ngrams)

    tf = sent_counts[ngram] / (sum(sent_counts.values))


    df = 0
    for image_id in ref_dict.keys():
        image_count = 0
        for ref_sent in ref_dict[image_id]:
            ref_ngrams = [ref_sent[i:i+n] for i in range(len(ref_sent) - n + 1)]
            ref_counts = col.Counter(ref_ngrams)
            image_count += ref_counts[ngram]
        if image_count == 0:
            image_count = 1
        df += image_count
            
    
    idf = np.log(len(ref_dict.keys()) / df)

    tfidf = tf * idf

    return tfidf

#def get_ngram_frequencies(ref_dict, max_n):
#    """
#    Get the number of occurences for each ngram in the reference sentences.
#
#    Parameters
#    ----------
#    ref_dict : dict([str])
#        Dictionary containing the reference sentences for each image.
#    max_n : int
#        Maximum size of ngrams to use.
#    """
#
#    freq_dict = dict()
#
#    sents = []
#    for l in ref_dict.values():
#        sents += l
#    
#    for n in list(range(max_n+1))[1:]:
#        freq_dict[n] = col.defaultdict(int)
#        for sent in sents:
#            ngrams = [tuple(sent[i:i+n]) for i in range(len(sent) - n + 1)]
#            for ngram in ngrams:
#                freq_dict[n][ngram] += 1
#
#    return freq_dict
            
        



    




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
        i += 1
        if i > 3:
            break
            

    matrix = DS_matrix(matrix_filename)

    for image_id, sent in orig_dict.items():
        decoded = matrix.reconstruct_sent(sent)
        decoded_dict[image_id] = nltk.word_tokenize(decoded)

    with open(decoded_filename, "wb") as decoded_file:
        pickle.dump(decoded_dict, decoded_file)

    with open(ref_filename, "wb") as ref_file:
        pickle.dump(ref_dict, ref_file)


decode_sentences("../cider/data/pascal50S.json", "../test_decoded.pkl", "../test_ref.pkl", "../matrix_1k/_matrix.pkl")
                 

#def calculate_ciderD(decoded_filename, ref_filename, log_filename, max_n=4):

#calculate_ciderD("../test_decoded.pkl", "../test_ref.pkl", "../test.log", max_n=4)
