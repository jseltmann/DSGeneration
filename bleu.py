import pickle
import json
from nltk.translate import sentence_bleu
import nltk

def bleu_pascal(orig_filename, decoded_filename, log_filename):
    """
    Calculate BLEU score for sentences from the pascal dataset.

    Parameters
    ----------
    orig_filename : str
        File containing the pascal50S dataset.
    decoded_filename : str
        File containing the decoded sentences.
    log_filename : str
        File to write the results to.
    """

    with open(orig_filename) as f:
        data = json.loads(f.read())

    orig_sents = dict()
    for entry in data:
        img_id = entry['image_id']
        if not img_id in orig_sents:
            orig_sents[img_id] = entry['caption']

    with open(decoded_filename, "rb") as decoded_file
        decoded_sents = pickle.load(decoded_file)

    scores = []
            
    for img_id in decoded_sents:
        orig = nltk.word_tokenize(orig_sents[img_id])
        decoded = nltk.word_tokenize(decoded_sents[img_id])

        score = sentence_bleu(orig, decoded)
        with open(log_filename, "a") as log_file:
            log_file.write(str(orig) + "\n")
            log_file.write(str(decoded) + "\n")
            log_file.write(str(score) + "\n\n")

        scores.append(score)

    avg_score = np.mean(scores)
    std_dev = np.std(scores)

    with open(log_filename, "a") as log_file:
        log_file.write("\n\n")
        log_file.write("average BLEU score: " + str(avg_score) + "\n")
        log_file.write("standard deviation: " + str(std_dev) + "\n")
        

