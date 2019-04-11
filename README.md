# DSGeneration

This repository contains code for the encoding of sentences into distributional vectors and the reconstruction of those sentences from the vectors.

See example.py for some examples of how to use them.

* Files that are part of the basic system:
  * build_bigram_matrix.py: code to train a model from a corpus
  * matrix_class.py: class containing a matrix
  * sowe2bow.py: bag-of-words reconstruction
* Files for testing:
  * bleu.py, cider.py: for sentence reordering
  * perplexity.py
  * sentence_positions.py: for positions of sentence and word vectors in the vector space
  * spearman_corr.py: calculate spearman correlation on MEN dataset or similar
  * test_bow_reconstruction.py: bag-of-words reconstruction

The file sowe2bow.py is a reimplementation of the code from White et. al. 2016, Generating Bags of Words from the Sums of their Word Embeddings. (https://white.ucc.asn.au/publications/White2016BOWgen/)

Contributor:
* Luca Ducceschi
* Johann Seltmann
* Aurelie Herbelot (not to the code, but to the project and the paper)
