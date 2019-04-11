from nltk.corpus import brown
from matrix_class import DS_matrix
import os

def sentence_bins(save_dir):
    """
    This generates the sentence bins for the testing of the bag-of-words
    reconstrution.

    Parameters
    ----------
    save_dir : str
        Directory to save the file to.
    """

    borders = [3,6,9,12,15,18,21,24]

    for i,l in enumerate(borders[:-1]):
        r = borders[i+1]
        sents = [s for s in brown.sents() if l <= len(s) < r]

        filename = os.join(save_dir, str(l) + "to" + str(r-1) + ".txt")

        with open(filename, "w") as outfile:
            for sent in sents:
                line = ' '.join(sent) + "\n"
            outfile.write(line)


def position_sents(sent_filename, sick_filename):
    """
    Generates the sentence file for the analysis of the positions in space.

    Parameters
    ----------
    sent_filename : str
        File to save the sentences to.
    sick_filename : str
        File containing the SICK dataset.
    """

    with open("SICK.txt") as sick_file:
        lines = sick_file.readlines()

    sents = set()
    for line in lines[1:]:
        sent = line.split('\t')[1]
        sents.add(sent)
        if len(sents) == 100:
            break

    with open(sent_filename, "w") as outfile:
        for sent in sents:
            outfile.write(sent)
            outfile.write("\n")

    # append sentences from brown corpus
    sents = [i for i in brown.sents() if len(i)>=3 and len(i)<=6 ]
    freqs = {}
    for sent in sents:
        for word in sent:
            if word.lower() in freqs:
                freqs[word.lower()] +=1
            else:
                freqs[word.lower()] = 1

    sorted_freqs = sorted(freqs.items(), key= lambda x : x[1], reverse=True)
    sorted_freqs = {i[0]:i[1]
                    for i in sorted_freqs if re.search("[a-z]+", i[0])}

    sents = [s for s in delete_unfrequent(sents, sorted_freqs, num_words=2000)]

    with open(sent_filename, "a") as outfile:
        for sent in to_write:
            outfile.write(sent)
            outfile.write("\n")

    # append longer sentences from brown corpus
    sents = [i for i in brown.sents() if len(i) > 6 and len(i) < 12]
    freqs = {}
    for sent in sents:
        for word in sent:
            if word.lower() in freqs:
                freqs[word.lower()] +=1
            else:
                freqs[word.lower()] = 1

    sorted_freqs = sorted(freqs.items(), key= lambda x : x[1], reverse=True)
    sorted_freqs = {i[0]:i[1]
                    for i in sorted_freqs if re.search("[a-z]+", i[0])}

    sents = [s for s in delete_unfrequent(sents, sorted_freqs, num_words=2000)]

    with open(sent_filename, "a") as outfile:
        for sent in to_write:
            outfile.write(sent)
            outfile.write("\n")
    


def delete_unfrequent(lofl,freq_list, num_words=1000):
    out_ = []
    lexicon = [i for i in freq_list.keys()][:num_words]
    for sent_ in lofl:
        temp_sent = [word_ for word_ in sent_ if word_ in lexicon]
        if len(temp_sent) < 3:
            continue
        else:
            out_.append(temp_sent)
    return out_
