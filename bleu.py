import pickle
import json
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk
import numpy as np
import collections as col


def bleu_pascal(ref_filename, orig_filename, decoded_filename, log_filename, num_bins=2):
    """
    Calculate BLEU score for sentences from the pascal dataset.

    Parameters
    ----------
    ref_filename : str
        File containing the reference sentences from the pascal50S dataset.
    orig_filename : str
        File containing the original sentences.
    decoded_filename : str
        File containing the decoded sentences.
    log_filename : str
        File to write the results to.
    num_bins : int
        Number of different bins of sentence lengths to compare.
    """

    with open(ref_filename, "rb") as f:
        ref_sents = pickle.load(f)#json.loads(f.read())
    with open(orig_filename, "rb") as f:
        orig_dict = pickle.load(f)

    len_counts = col.defaultdict(int)
    for img_id in orig_dict:
        sent = nltk.word_tokenize(orig_dict[img_id])
        len_counts[len(sent)] += 1

    generated_bins = 0
    len_to_bin = dict()
    bin_to_len = dict()
    sent_num = len(orig_dict)
    bin_size = sent_num / num_bins

    curr_sum = 0
    curr_bin = 0
    bin_sum = dict()
    for length in sorted(len_counts):
        curr_sum += len_counts[length]
        len_to_bin[length] = curr_bin
        if curr_bin in bin_to_len:
            bin_to_len[curr_bin].append(length)
        else:
            bin_to_len[curr_bin] = [length]
        if curr_sum >= bin_size and generated_bins < num_bins:
            bin_sum[curr_bin] = curr_sum
            curr_sum = 0
            curr_bin += 1
            generated_bins += 1
    bin_sum[curr_bin] = curr_sum

    scores = dict()

    with open(decoded_filename, "rb") as decoded_file:
        decoded_sents = pickle.load(decoded_file)


    references_complete = dict()
    decoded_complete = dict()
    
    for img_id in decoded_sents:
        decoded = decoded_sents[img_id]

        references = ref_sents[img_id]

        score = sentence_bleu(references, decoded)
        with open(log_filename, "a") as log_file:
            log_file.write(str(orig_dict[img_id]) + "\n")
            log_file.write(str(decoded) + "\n")
            log_file.write(str(score) + "\n\n")

        orig_sent = nltk.word_tokenize(orig_dict[img_id])
        bin_num = len_to_bin[len(orig_sent)]
        if bin_num in scores:
            scores[bin_num].append(score)
            references_complete[bin_num].append(references)
            decoded_complete[bin_num].append(decoded)
        else:
            scores[bin_num] = [score]
            references_complete[bin_num] = [references]
            decoded_complete[bin_num] = [decoded]

    for bin_num in range(num_bins):
        avg = np.mean(scores[bin_num])
        stddev = np.std(scores[bin_num])
        corpus_score = corpus_bleu(references_complete[bin_num],
                                   decoded_complete[bin_num])

        len_min = min(bin_to_len[bin_num])
        len_max = max(bin_to_len[bin_num])

        with open(log_filename, "a") as log_file:
            log_file.write("\n\n")
            log_file.write("sentence lengths: " + str(len_min)
                           + " to " + str(len_max) + "\n")
            log_file.write("number of sentences: "
                           + str(bin_sum[bin_num]) + "\n")
            log_file.write("average sentence BLEU score: " + str(avg) + "\n")
            log_file.write("standard deviation: " + str(stddev) + "\n")
            log_file.write("corpus BLEU score: " + str(corpus_score) + "\n")

