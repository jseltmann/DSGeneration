from matrix_class import DS_matrix
import sowe2bow as s2b
import nltk
import numpy as np
from copy import copy
from datetime import datetime
import multiprocessing as mp

def decode_sents(sents_filename, matrix_filename, log_filename, skipped_filename, num_words=None, timeout=30, sent_num=500):
    """
    Encode and decode sentences and write the resultinig bag of words to a log_filename.
    """

    matrix = DS_matrix(matrix_filename)

    if num_words is not None:
        vocab = list(matrix.vocab_order.keys())[:num_words]

        matrix = matrix.less_words_matrix(vocab)
    else:
        vocab = list(matrix.vocab_order.keys())


    def call_greedy_search(matrix, target, queue):
        res_sent, _ = s2b.greedy_search(matrix,target)
        queue.put(res_sent)
        
    decoded_count = 0

    for i, line in enumerate(open(sents_filename)):
        print(i)
        print(line)
        print(datetime.now())
        target = matrix.encode_sentence(line)
        #res_sent, score = s2b.greedy_search(matrix, target)

        output = mp.Queue()
        p = mp.Process(target=call_greedy_search,
                       name="call_greedy_search",
                       args=(matrix, target, output))

        p.start()
        p.join(timeout)

        if p.is_alive():
            res_sent = None
            p.terminate()
            with open(skipped_filename, "a") as skipped_file:
                skipped_file.write(line)
            continue
        else:
            res_sent = output.get()
            decoded_count += 1

            with open(log_filename, "a") as log_file:
                sent = list(map((lambda x:x.lower()), nltk.word_tokenize(line)))
                line = str(i) + " || "
                for word in sent:
                    line += word + " "
                line += "|| "
                if res_sent is not None:
                    for word in res_sent:
                        line += word + " "
                line += "\n"

                log_file.write(line)

            if decoded_count >= sent_num:
                break


def evaluate_decoding(results_filename, log_filename):
    """
    Test the decoding of sentence vectors with the greedy search from White et.al.

    Parameters
    ----------
    results_filename : str
        File containing original and decoded sentences.
    log_filename : str
        File to write log to.
    """

    #matrix = DS_matrix(matrix_filename)

    #if num_words is not None:
    #    vocab = list(matrix.vocab_order.keys())[:num_words]

    #    matrix = matrix.less_words_matrix(vocab)
    #else:
    #    vocab = list(matrix.vocab_order.keys())


    #get differences between original and reconstruction both for original sentence
    #and for the original sentence limited to words in the matrix
    diff_sum_ori = 0
    diff_count_ori = 0

    exact_match_count_ori = 0
    total_count_ori = 0

    len_diffs_ori = []
    nums_removed_ori = []
    nums_added_ori = []

    incorrect_pairs_ori = []


    diff_sum_lim = 0
    diff_count_lim = 0

    exact_match_count_lim = 0
    total_count_lim = 0

    len_diffs_lim = []
    nums_removed_lim = []
    nums_added_lim = []

    incorrect_pairs_lim = []


    def check_same(sent1, sent2):
        sent2_copy = copy(sent2)
        for w1 in sent1:
            if not w1 in sent2_copy:
                return False
            sent2_copy.remove(w1)
        if len(sent2_copy) > 0:
            return False
        return True

    for i, line in enumerate(open(results_filename)):
        #if i == 10:
        #    print(i)
        #if i % 100 == 0:
        #    print(i)
        print(i)
        print(line)
        print(datetime.now())
        #target = matrix.encode_sentence(line)
        #res_sent, score = s2b.greedy_search(matrix, target)

        #with open(log_filename, "a") as log_file:
        #    sent = list(map((lambda x:x.lower(), nltk.word_tokenize(line))))
        #    line = str(i) + " || "
        #    for word in sent:
        #        line += word + " "
        #    line += "|| "
        #    for word in res_sent:
        #        line += word + " "
        #    line += "\n"

        #    log_file.write(line)
            
        _, orig_sent, res_sent = line.split("||")

        res_sent = nltk.word_tokenize(res_sent)

        diff_sum_ori += score
        diff_count_ori += 1

        corr_sent = [w.lower() for w in nltk.word_tokenize(orig_sent)]
        if check_same(corr_sent, res_sent):
            exact_match_count_ori += 1
        else:
            num_removed = len([w for w in corr_sent if w not in res_sent])
            num_added = len([w for w in res_sent if w not in corr_sent])
            len_diff = len(corr_sent) - len(res_sent)
            nums_removed_ori.append(num_removed)
            nums_added_ori.append(num_added)
            len_diffs_ori.append(len_diff)
            incorrect_pairs_ori.append((orig_sent, res_sent, num_removed, num_added, len_diff))
            
        total_count_ori += 1

        #same values, but with the sentence limited to words appearing in the matrix
        diff_sum_lim += score
        diff_count_lim += 1

        corr_sent = [w for w in corr_sent if w in vocab]
        if check_same(corr_sent, res_sent):
            exact_match_count_lim += 1
        else:
            num_removed = len([w for w in corr_sent if w not in res_sent])
            num_added = len([w for w in res_sent if w not in corr_sent])
            len_diff = len(corr_sent) - len(res_sent)
            nums_removed_lim.append(num_removed)
            nums_added_lim.append(num_added)
            len_diffs_lim.append(len_diff)
            incorrect_pairs_lim.append((orig_sent, res_sent, num_removed, num_added, len_diff))
            
        total_count_lim += 1


    with open(log_filename, "a") as log_file:
        log_file.write("\n\n\n\n")
        log_file.write("Incorrectly decoded pairs for original sentences\nCorrect sentence first, decoded second\n\n")

        for corr, incorr, n_removed, n_added, len_diff in incorrect_pairs_ori:
            log_file.write(str(corr) + "\n")
            log_file.write(str(incorr) + "\n")
            log_file.write("#removed words: " + str(n_removed) + "\n")
            log_file.write("#added words: " + str(n_added) + "\n")
            log_file.write("#length difference: " + str(len_diff) + "\n")
            log_file.write("\n\n")

        log_file.write("\n\n")

        log_file.write("Incorrectly decoded pairs for limited sentences\nCorrect sentence first, decoded second\n\n")

        for corr, incorr, n_removed, n_added, len_diff in incorrect_pairs_lim:
            log_file.write(str(corr))
            log_file.write(str(incorr))
            log_file.write("\n#removed words: " + str(n_removed) + "\n")
            log_file.write("#added words: " + str(n_added) + "\n")
            log_file.write("#length difference: " + str(len_diff) + "\n")
            log_file.write("\n\n")

        log_file.write("\n\n")


        log_file.write("statistic using original sentences:\n")
        if diff_count_ori != 0:
            avg = diff_sum_ori / diff_count_ori
            log_file.write("average distance: " + str(avg) + "\n")
        log_file.write("exact matches: " + str(exact_match_count_ori) + " out of " + str(total_count_ori) + "\n")

        perc_correct = exact_match_count_ori / total_count_ori
        log_file.write("fraction of exact matches: " + str(perc_correct) + "\n")

        log_file.write("\nThe following values don't consider exact matches.\n")
        if len(nums_removed_ori) != 0:
            avg_removed = np.average(nums_removed_ori)
            var_removed = np.var(nums_removed_ori)
            log_file.write("avg number of removed words: " + str(avg_removed) + " variance: " + str(var_removed) + "\n")

        if len(nums_added_ori) != 0:
            avg_added = np.average(nums_added_ori)
            var_added = np.var(nums_added_ori)
            log_file.write("avg number of added words: " + str(avg_added) + " variance: " + str(var_added) + "\n")

        if len(len_diffs_ori) != 0:
            avg_len_diff = np.average(len_diffs_ori)
            var_len_diff = np.var(len_diffs_ori)
            log_file.write("avg length difference: " + str(avg_len_diff) + " variance: " + str(var_len_diff) + "\n")

        log_file.write("\n\nstatistic using limited sentences:\n")
        if diff_count_lim != 0:
            avg = diff_sum_lim / diff_count_lim
            log_file.write("average distance: " + str(avg) + "\n")
        log_file.write("exact matches: " + str(exact_match_count_lim) + " out of " + str(total_count_lim) + "\n")

        perc_correct = exact_match_count_lim / total_count_lim
        log_file.write("fraction of exact matches: " + str(perc_correct) + "\n")

        log_file.write("\nThe following values don't consider exact matches.\n")
        if len(nums_removed_lim) != 0:
            avg_removed = np.average(nums_removed_lim)
            var_removed = np.var(nums_removed_lim)
            log_file.write("avg number of removed words: " + str(avg_removed) + " variance: " + str(var_removed) + "\n")

        if len(nums_added_lim) != 0:
            avg_added = np.average(nums_added_lim)
            var_added = np.var(nums_added_lim)
            log_file.write("avg number of added words: " + str(avg_added) + " variance: " + str(var_added) + "\n")

        if len(len_diffs_lim) != 0:
            avg_len_diff = np.average(len_diffs_lim)
            var_len_diff = np.var(len_diffs_lim)
            log_file.write("avg length difference: " + str(avg_len_diff) + " variance: " + str(var_len_diff) + "\n")


        
        

#test_decoding("../test_sents.txt", "../matrix_50k/_matrix.pkl", "../test.log", num_words=10000)#, words_filename="sents_from_brown/5000_words.txt")
#test_decoding("brown_sents/1000_freq_sents_from_brown.txt", "../matrix_50k/_matrix.pkl", "../50k_1000_short_sents.log")

decode_sents("../brown_sents_bins_incl_non_matrix/15to17.txt", "../matrix_50k/_matrix.pkl", "../decoded_500_50k_15to17.log", "../skipped_50k_15to17.log", timeout=900)
#decode_sents("../brown_sents_bins_incl_non_matrix/18to20.txt", "../matrix_50k/_matrix.pkl", "../decoded_10k_18to20.log", num_words=10000, timeout=900)
