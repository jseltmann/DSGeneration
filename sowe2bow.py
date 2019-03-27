"""
This module translates sowe2bow from
https://white.ucc.asn.au/publications/White2016BOWgen/ into python,
i.e. it implements the bag-of-words reconstruction algorithm.
The main method is greedy_search().

Attributes
----------
epsilon : float
    constant used for round off when comparing scores
    -- numbers whose difference is less than ϵ are considered the same
"""


import numpy as np
from copy import copy

epsilon = 10.0 ** -5


def get_end(LL, ws):
    """
    Add the vectors of the words represented by ws.

    Parameters
    ----------
    LL : DS_matrix
        Matrix containing the vectors.
    ws : [String]
        Current bag of words.

    Return
    ------
    sofar : np.ndarray
        Sum of the vectors.
    """
    if len(ws) == 0:
        shape = (1, LL.get_vector(list(LL.vocab_order.keys())[0]).shape[1])
        sofar = np.zeros(shape)
    else:
        vectors = []
        for word in ws:
            vectors.append(LL.get_vector(word))

        sofar = np.sum(vectors, axis=0)

    return sofar


def score_possible_additions(LL, target, end_point):
    """
    Score all possible additions.

    Parameters
    ----------
    LL : DS_matrix
        Matrix containing the vectors.
    target : np.ndarray
        Vector representing the encoded sentence.
    end_point : np.ndarray
        Vector representing the current bag of words.

    Return
    ------
    word_scores : dict
        For each word in LL, the score
        that would be reached by adding that word.
    """

    diff = end_point - target

    word_scores = dict()
    matrix = LL.matrix.tocsr()

    for i, word in enumerate(LL.get_words()):
        pos = LL.vocab_order[word]

        vec = matrix[pos].toarray()
        vec += diff
        vec = vec * vec
        s = np.sum(vec)
        score = -np.sqrt(s)

        word_scores[word] = score

    return word_scores


def fitness(target, end_point):
    """
    Evaluates distance from the end_point vector to the target vector.

    Parameters
    ----------
    target : np.ndarray
        The vector representing the encoded sentence.
    end_point : np.ndarray
        The sum of the vectors currently in the bag.

    Return
    ------
    fitness : float
        Distance between target and end_point, multiplied by -1.
    """

    diff = end_point - target
    diff = diff * diff

    fitness = - np.sum(diff)

    return fitness


def greedy_addition(LL, target, initial_word_set, max_additions=float("inf")):
    """
    Perform the greedy addition step to find
    the bag of words from a sum of word embeddings.

    Parameters
    ----------
    LL : DS_matrix
        Matrix containing the vectors representing the words.
    target : np.ndarray
        Vector that represents the encoded sentence.
    initial_word_set : [String]
        Initial bag of words.
    max_additions : float
        Maximim number of additions to perform. Usually infinite.

    Return
    ------
    best_word_set : [String]
        Best list of words found.
    best_score : float
        Score of best_word_set. Scores are <= 0, higher scores are better.
    """

    best_word_set = copy(initial_word_set)
    end_point = get_end(LL, best_word_set)
    best_score = fitness(target, end_point)

    curr_additions = 0
    while curr_additions < max_additions:
        curr_additions += 1

        addition_scores = score_possible_additions(LL, target, end_point)
        addition = max(addition_scores.keys(),
                       key=(lambda x: addition_scores[x]))
        addition_score = addition_scores[addition]

        if addition_score > best_score + epsilon:
            best_score = addition_score
            best_word_set.append(addition)
            end_point += LL.get_vector(addition)
        else:
            break

    return best_word_set, best_score


def word_swap_refinement(LL, target, initial_word_set):
    """
    Perform the 1-substitution step to refine the bag of words.

    Parameters
    ----------
    LL : DS_matrix
        Matrix containing the vectors representing the words.
    target : np.ndarray
        Vector that represents the encoded sentence.
    initial_word_set : [String]
        Initial bag of words.

    Return
    ------
    best_word_set : [String]
        Best list of words found.
    best_score : float
        Score of best_word_set. Scores are <= 0, higher scores are better.
    """

    best_word_set = copy(initial_word_set)
    end_point = get_end(LL, initial_word_set)
    best_score = fitness(target, end_point)

    for i, word in enumerate(initial_word_set[:-1]):
        word_set = copy(initial_word_set)
        word_set.pop(i)

        sub_endpoint = end_point - LL.get_vector(word)
        subset_score = fitness(target, sub_endpoint)
        if subset_score > best_score + epsilon:
            best_score = subset_score
            best_word_set = word_set

        add_word_set, add_score = greedy_addition(LL, target, word_set, 1)
        if add_score > best_score + epsilon:
            best_score = add_score
            best_word_set = add_word_set

    return best_word_set, best_score


def greedy_search(LL, target, rounds=1000, log=False):
    """
    Perform greedy search and 1-substitution, repeated until convergence.

    Parameters
    ----------
    LL : DS_matrix
        Matrix containing the vectors representing the words.
    target : np.ndarray
        Vector that represents the encoded sentence.
    rounds : int
        Maximum number of rounds of Greedy addition and 1-substitution.
        Set to low value for debugging.
    log : bool
        Display logging messages.

    Return
    ------
    word_list : [String]
        Bag of words to decode the target sentence.
    best_score : float
        Score of word_list. Scores are <= 0, with higher scores being better.
    """

    word_list = []
    best_word_list = copy(word_list)
    best_score = float("-inf")

    for round in range(rounds):
        word_list, add_score = greedy_addition(LL, target, word_list)

        if log:
            print("POST_ADD_STEP:", add_score, word_list)

        best_word_list = word_list

        if add_score >= 0:
            best_score = add_score
            break

        word_list, swap_score = word_swap_refinement(LL, target, word_list)

        if log:
            print("POST_SWAP_STEP:", swap_score, word_list)

        best_word_list = word_list

        converged = (best_score - epsilon < swap_score
                     and swap_score < best_score + epsilon)
        best_score = swap_score

        if converged:
            break

    return word_list, best_score
