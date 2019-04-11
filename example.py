# Example file for the DS matrices.
# Note that execution will fail if you
# don't have the respective corpora in place.

import build_bigram_matrix as bbm
from matrix_class import DS_matrix
import sowe2bow as s2b
import test_bow_reconstruction as tbr
import cider as ci


# train a matrix:
bbm.build_ngram_probability_matrix("../some_corpus.txt",
                                   "../frequency_distribution.txt",
                                   "../matrix_file.pkl",
                                   num_words=50000)
# The corpus should be pre-tokenized and contain one sentence per line.
# The frequency distribution can be generated from the corpus with:
# tr "\t" "\n" < some_corpus.txt | sort | uniq -c | sort -nr > frequency_distribution.txt


# encode a sentence
matrix = DS_matrix("../matrix_file.pkl")
sentence_vector = matrix.encode_sentence("The cat climbs the tree.")
# reconstruct the words in the sentence
words, _ = s2b.greedy_search(matrix, sentence_vector)


# generate a matrix with a smaller number of words to speed up the search
# note that the resulting matrix isn't square anymore,
# and it's entries don't correspond to bigram probabilities
word_list = matrix.get_words()[:10000]
matrix2 = matrix.less_words_matrix(word_list)


# the inner representation of the matrix is a scipy.sparse.lil_matrix
# for some operations, a different representation can be useful
# for example, if you need to retrieve many word vector,
# the csr_matrix is more efficient:
matrix.tocsr()



# test bag-of-words reconstruction

## first: decode sentences to test
tbr.decode_sents("../test_sents.txt", "../matrix_file.pkl",
                 "../decoded_sents.txt", "../skipped_sents.txt",
                 num_words=10000,
                 timeout=900, sent_num=500)
# The timeout exists because some sentences take very long to en- and decode.
# If a sentence takes longer than that number of seconds, the function skips it.
# sent_num is the overall number of sentences from the test_sents file to be used.
# num_words is the number of vectors from the original matrix to use.
# set it to less than the actual number for faster reconstruction.

## second: evaluate the decoding
tbr.evaluate_decoding("../decoded_sents.txt", "evaluated_decoding.txt",
                      "../matrix_file.pkl", num_words=10000)
# evaluated_decoding.txt will contain some statistics for each sentence
# and overall statistics at the end.
# num_words should be set to the same value as for decode_sents().



# test reconstruction and reordering on the Pascal50S dataset.
ci.decode_sentences("../datafile.json", "../decoded_sents.pkl",
                    "../reference_sents.pkl", "../original_sents.pkl",
                    "../matrix_file.pkl")
ci.calculate_ciderD("../decoded_sents.pkl", "../reference_sents.pkl",
                    "../original_sents.pkl", "../cider_evaluation.log",
                    num_bins=1)
# num_bins is used to distribute the sentences into bins according to their length.
# Use 1 if you just want an evaluation on the whole set.
