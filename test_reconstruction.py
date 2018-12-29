from matrix_class import DS_matrix
import sowe2bow as s2b
import nltk

def test_decoding(sents_filename, matrix_filename, log_filename, words_filename=None):
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
    words_filename : str
        Optional, 
        file containing words to which to limit the 
        matrix in order to speed up calculations.
    """

    matrix = DS_matrix(matrix_filename)

    if words_filename is not None:
        words = set()
        for line in open(words_filename):
            words.add(line.strip())

        matrix = matrix.less_words_matrix(words)


    diff_sum = 0
    diff_count = 0

    exact_match_count = 0
    total_count = 0

    incorrect_pairs = []
    
    for i, line in enumerate(open(sents_filename)):
        if i == 10:
            print(i)
        if i % 100 == 0:
            print(i)
        target = matrix.encode_sentence(line)
        res_sent, score = s2b.greedy_search(matrix, target)

        with open(log_filename, "a") as log_file:
            log_file.write("sentence: " + str(line) + "\n")
            log_file.write("decoded: " + str(res_sent) + "\n")
            log_file.write("score: " + str(score) + "\n\n")

        diff_sum += score
        diff_count += 1

        words = nltk.word_tokenize(line)
        if set(words) == set(res_sent):
            exact_match_count += 1
        else:
            incorrect_pairs.append((line, res_sent))
        total_count += 1


    with open(log_filename, "a") as log_file:
        log_file.write("\n\n\n\n")
        log_file.write("Incorrectly decoded pairs\nCorrect sentence first, decoded second\n\n")

        for corr, incorr in incorrect_pairs:
            log_file.write(str(corr))
            log_file.write(str(incorr))
            log_file.write("\n\n")

        log_file.write("\n\n")
        
        if diff_count != 0:
            avg = diff_sum / diff_count
            log_file.write("average distance: " + str(avg) + "\n")
        log_file.write("exact matches: " + str(exact_match_count) + " out of " + str(total_count))
        

test_decoding("sents_from_brown/2000_freq_long_sents.txt", "../matrix_50k/_matrix.pkl", "../2000_sents_long_5000_words.log", words_filename="sents_from_brown/5000_words.txt")
#test_decoding("brown_sents/1000_freq_sents_from_brown.txt", "../matrix_50k/_matrix.pkl", "../50k_1000_short_sents.log")
