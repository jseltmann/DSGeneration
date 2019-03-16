import nltk
import re
from nltk.corpus import brown


#sents = [i for i in brown.sents() if len(i)>=3 and len(i)<=6 ]
sents = [i for i in brown.sents() if len(i)>=12 ]
freqs = {}

for sent in sents:
    for word in sent:
        if word.lower() in freqs:
            freqs[word.lower()] +=1
        else:
            freqs[word.lower()] = 1
            
sorted_freqs = sorted(freqs.items(), key= lambda x : x[1], reverse=True)
sorted_freqs = {i[0]:i[1] for i in sorted_freqs if re.search("[a-z]+", i[0])}


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

##with open("/home/luca/Data/1000_freq_sents_from_brown.txt", "w") as write_out:
#with open("brown_sents/1000_freq_sents_from_brown.txt", "w") as write_out:
#    
#    for i in delete_unfrequent(sents, sorted_freqs, num_words=1000):
#        write_out.write("\t".join(i) + "\n")
#
#
#with open("brown_sents/1500_freq_sents_from_brown.txt", "w") as write_out:
#    
#    for i in delete_unfrequent(sents, sorted_freqs, num_words=1500):
#        write_out.write("\t".join(i) + "\n")

#with open("brown_sents/1000_freq_long_sents.txt", "w") as write_out:
#    
#    count_long_sents = 0
#    for i in delete_unfrequent(sents, sorted_freqs, num_words=1000):
#        if len(i) > 5:
#            write_out.write("\t".join(i) + "\n")
#            count_long_sents += 1
#        if count_long_sents >= 1075:
#            break


with open("sents_from_brown/2000_freq_very_long_sents.txt", "w") as write_out:
    
    count_long_sents = 0
    for i in delete_unfrequent(sents, sorted_freqs, num_words=2000):
        if len(i) > 12:
            write_out.write("\t".join(i) + "\n")
            count_long_sents += 1
        if count_long_sents >= 1075:
            break
