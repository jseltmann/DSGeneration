from matrix_class import DS_matrix
import sowe2bow as s2b
import nltk
import numpy as np

def test_decoding(sents_filename, matrix_filename, log_filename, num_words=None):
    """
    Test the decoding of sentence vectors with the greedy search from White et.al.

    Parameters
    ----------
    sents_file : str
        Filename of a file containing test sentences.
    matrix_filename : str
        Filename of the file containing the matrix.
    log_filename : str
        File to write log to.
    num_words : int
        Optional, 
        number of words to which to limit the 
        matrix in order to speed up calculations.
    """

    matrix = DS_matrix(matrix_filename)

    if num_words is not None:
        vocab = list(matrix.vocab_order.keys())[:num_words]

        matrix = matrix.less_words_matrix(vocab)
    else:
        vocab = list(matrix.vocab_order.keys())


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
        for w1 in sent1:
            if not w1 in sent2:
                return False
            sent2.remove(w1)
        if len(sent2) > 0:
            return False
        return True

    for i, line in enumerate(open(sents_filename)):
        if i == 10:
            print(i)
        if i % 100 == 0:
            print(i)
        target = matrix.encode_sentence(line)
        res_sent, score = s2b.greedy_search(matrix, target)


        diff_sum_ori += score
        diff_count_ori += 1

        corr_sent = [w.lower() for w in nltk.word_tokenize(line)]
        if check_same(corr_sent, res_sent):
            exact_match_count_ori += 1
        else:
            num_removed = len([w for w in corr_sent if w not in res_sent])
            num_added = len([w for w in res_sent if w not in corr_sent])
            len_diff = len(corr_sent) - len(res_sent)
            nums_removed_ori.append(num_removed)
            nums_added_ori.append(num_added)
            len_diffs_ori.append(len_diff)
            incorrect_pairs_ori.append((line, res_sent, num_removed, num_added, len_diff))
            
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
            incorrect_pairs_lim.append((line, res_sent, num_removed, num_added, len_diff))
            
        total_count_lim += 1


    with open(log_filename, "a") as log_file:
        log_file.write("\n\n\n\n")
        log_file.write("Incorrectly decoded pairs for original sentences\nCorrect sentence first, decoded second\n\n")

        for corr, incorr, n_removed, n_added, len_diff in incorrect_pairs_ori:
            log_file.write(str(corr))
            log_file.write(str(incorr))
            log_file.write("#removed words: " + str(n_removed) + "\n")
            log_file.write("#added words: " + str(n_added) + "\n")
            log_file.write("#length difference: " + str(len_diff) + "\n")
            log_file.write("\n\n")

        log_file.write("\n\n")

        log_file.write("Incorrectly decoded pairs for limited sentences\nCorrect sentence first, decoded second\n\n")

        for corr, incorr, n_removed, n_added, len_diff in incorrect_pairs_lim:
            log_file.write(str(corr))
            log_file.write(str(incorr))
            log_file.write("#removed words: " + str(n_removed) + "\n")
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


        
        

#test_decoding("../test_sents.txt", "../matrix_1k/_matrix.pkl", "../test.log", num_words=500)#, words_filename="sents_from_brown/5000_words.txt")
#test_decoding("brown_sents/1000_freq_sents_from_brown.txt", "../matrix_50k/_matrix.pkl", "../50k_1000_short_sents.log")

print("Test decoding of brown sents on 10k matrix, excluding words not in matrix")
test_decoding("../brown_sents_bins_excl_non_10k/3to5.txt", "../matrix_50k/_matrix.pkl", "../bins_excl_non_10k_logs/3to5.log", num_words=10000)
print("tested sentences of lengths 3-5")
test_decoding("../brown_sents_bins_excl_non_10k/6to8.txt", "../matrix_50k/_matrix.pkl", "../bins_excl_non_10k_logs/6to8.log", num_words=10000)
print("tested sentences of lengths 6-8")
test_decoding("../brown_sents_bins_excl_non_10k/9to11.txt", "../matrix_50k/_matrix.pkl", "../bins_excl_non_10k_logs/9to11.log", num_words=10000)
print("tested sentences of lengths 9-11")
test_decoding("../brown_sents_bins_excl_non_10k/12to14.txt", "../matrix_50k/_matrix.pkl", "../bins_excl_non_10k_logs/12to14.log", num_words=10000)
print("tested sentences of lengths 12-14")
test_decoding("../brown_sents_bins_excl_non_10k/15to17.txt", "../matrix_50k/_matrix.pkl", "../bins_excl_non_10k_logs/15to17.log", num_words=10000)
print("tested sentences of lengths 15-17")
test_decoding("../brown_sents_bins_excl_non_10k/18to20.txt", "../matrix_50k/_matrix.pkl", "../bins_excl_non_10k_logs/18to20.log", num_words=10000)
print("tested sentences of lengths 21-24")
test_decoding("../brown_sents_bins_excl_non_10k/21to24.txt", "../matrix_50k/_matrix.pkl", "../bins_excl_non_10k_logs/21to24.log", num_words=10000)
print("tested sentences of lengths 21-24")

