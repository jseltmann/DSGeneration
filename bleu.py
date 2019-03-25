import pickle
import json
from nltk.translate.bleu_score import sentence_bleu
import nltk
import numpy as np


def bleu_pascal(ref_filename, decoded_filename, log_filename):
    """
    Calculate BLEU score for sentences from the pascal dataset.

    Parameters
    ----------
    ref_filename : str
        File containing the reference sentences from the pascal50S dataset.
    decoded_filename : str
        File containing the decoded sentences.
    log_filename : str
        File to write the results to.
    """

    with open(ref_filename) as f:
        data = json.loads(f.read())

    ref_sents = dict()
    for entry in data:
        img_id = entry['image_id']
        ref = nltk.word_tokenize(entry['caption'])
        ref = list(map(lambda x: x.lower(), ref))
        if img_id not in ref_sents:
            ref_sents[img_id] = [ref]
        else:
            ref_sents[img_id].append(ref)

    with open(decoded_filename, "rb") as decoded_file:
        decoded_sents = pickle.load(decoded_file)

    scores = []

    i = 0
    for img_id in decoded_sents:
        i += 1
        decoded = decoded_sents[img_id]

        references = ref_sents[img_id]

        score = sentence_bleu(references, decoded)
        with open(log_filename, "a") as log_file:
            log_file.write(str(references[0]) + "\n")
            log_file.write(str(decoded) + "\n")
            log_file.write(str(score) + "\n\n")

        scores.append(score)

    avg_score = np.mean(scores)
    std_dev = np.std(scores)

    with open(log_filename, "a") as log_file:
        log_file.write("\n\n")
        log_file.write("average BLEU score: " + str(avg_score) + "\n")
        log_file.write("standard deviation: " + str(std_dev) + "\n")
