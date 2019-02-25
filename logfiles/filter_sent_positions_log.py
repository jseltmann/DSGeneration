import numpy as np
import nltk
import re

sent_dist_list = []
word_dist_list = []

next_sent_dist = False
next_word_dist = False

with open("positions_in_space/sent_positions.log") as log:
    for line in log:
        if len(nltk.word_tokenize(line)) > 1:
            next_sent_dist = True
            print(line)
            print("next_sent")
            continue
        elif len(nltk.word_tokenize(line)) == 1 and not next_sent_dist and not next_word_dist:
            print(line)
            print("next_word")
            next_word_dist = True
            continue
        else:
            if next_sent_dist:
                print(line)
                sent_dist_list.append(float(line))
                next_sent_dist=False
                continue
            if next_word_dist:
                print(line)
                word_dist_list.append(float(line))
                next_word_dist=False
                continue

sent_avg_dist = np.mean(sent_dist_list)
sent_var_dist = np.var(sent_dist_list)
                 
print(sent_avg_dist)
print(sent_var_dist)

word_avg_dist = np.mean(word_dist_list)
word_var_dist = np.var(word_dist_list)

print(word_avg_dist)
print(word_var_dist)
