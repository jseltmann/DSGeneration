from nltk.corpus import brown
from matrix_class import DS_matrix

borders = [3,6,9,12,15,18,21,24]


matrix = DS_matrix("../matrix_50k/_matrix.pkl")
vocabulary = set(list(matrix.vocab_order.keys())[:50000])

def rm_uncommon(sent):
  words = [w for w in sent if w not in vocabulary]
  return words

for i,l in enumerate(borders[:-1]):
  r = borders[i+1]

  sents = [rm_uncommon(s) for s in brown.sents() if l < len(rm_uncommon(s)) < r]

  filename = "../brown_sents_bins_excl_non_50k/" + str(l) + "to" + str(r) + ".txt"

  with open(filename, "w") as outfile:
    for sent in sents[:1000]:
      line = ' '.join(sent) + "\n"
      outfile.write(line)



