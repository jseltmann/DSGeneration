import pickle
import numpy as np
from scipy.spatial.distance import cosine
import random

with open("../rev_matrices/bigram_matrix_dict_complete.pkl", "rb") as matrix_file:
    matrix = pickle.load(matrix_file)
with open("../rev_matrices/bigram_matrix_dict_complete.pkl_order", "rb") as order_file:
    vec_order = pickle.load(order_file)



stopwords = [
    "", "(", ")", "a", "about", "an", "and", "are", "around", "as", "at",
    "away", "be", "become", "became", "been", "being", "by", "did", "do",
    "does", "during", "each", "for", "from", "get", "have", "has", "had", "he",
    "her", "his", "how", "i", "if", "in", "is", "it", "its", "made", "make",
    "many", "most", "not", "of", "on", "or", "s", "she", "some", "that", "the",
    "their", "there", "this", "these", "those", "to", "under", "was", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your"
]


def get_avg(words, ignore=[]):
    sumwords = None
    n = 0
    for word in words:
        if not word in matrix or word in ignore:
            continue

        n += 1
        if sumwords is None:
            sumwords = matrix[word]
            continue

        sumwords = sumwords + matrix[word]

    avg = sumwords / n
    return avg

#sw_diffs = []
#for word in stopwords:
#    if not word in matrix:
#        continue
#    vec = matrix[word]
#
#    vec_min = np.amin(vec)
#    vec_max = np.amax(vec)
#    if vec_max - vec_min > 0.6:
#        print(word, vec_max-vec_min)
#
#    sw_diffs.append(vec_max-vec_min)
#    #print(word, vec_max-vec_min)
#
#print()
#print()
#print()
#    
#random_non_stop = []
#while(len(random_non_stop) < len(stopwords)):
#    word = random.choice(list(matrix.keys()))
#    if not word in stopwords:
#        random_non_stop.append(word)
#
#n_sw_diffs = []
#for word in random_non_stop:
#    if not word in matrix:
#        continue
#    vec = matrix[word]
#
#    vec_min = np.amin(vec)
#    vec_max = np.amax(vec)
#
#    if vec_max - vec_min > 0.6:
#        print(word, vec_max-vec_min)
#    n_sw_diffs.append(vec_max - vec_min)
    #print(word, vec_max-vec_min)

#print(min(sw_diffs), max(sw_diffs))
#print(min(n_sw_diffs), max(n_sw_diffs))

#stop_avg = get_avg(stopwords)    
#matrix_avg = get_avg(matrix)
#non_stop_avg = get_avg(matrix, ignore=stopwords)

#print("cosine similarity:")
#
#print("stopwords to global average")
#similarity = 1 - cosine(stop_avg, matrix_avg)
#print(similarity)
#
#print("stopwords to all non-stopwords")
#similarity = 1 - cosine(stop_avg, non_stop_avg)
#print(similarity)

#get random non-stopwords
#for _ in range(10):
#    random_non_stop = []
#    while(len(random_non_stop) < len(stopwords)):
#        word = random.choice(list(matrix.keys()))
#        if not word in stopwords:
#            random_non_stop.append(word)
#
#    print(random_non_stop)
#
#    avg = get_avg(random_non_stop)
#    print("random non-stopwords to stopwords")

#for word in matrix:#stopwords:
#    if not word in matrix:
#        continue
#    similarity = 1 - cosine(stop_avg, matrix[word])
#    if similarity > 0.5:
#        print(word, similarity)

    #similarity = 1 - cosine(stop_avg, matrix[word])
    #if similarity < 0.3:
    #    print(word, similarity)
    
#for word in stopwords:
#    if not word in matrix:
#        continue
#
#    bins = [0,0,0,0]
#    
#    for entry in matrix[word]:
#        if entry <= 0.25:
#            bins[0] += 1
#        elif entry <= 0.5:
#            bins[1] += 1
#        elif entry <= 0.75:
#            bins[2] += 1
#        else:
#            bins[3] += 1
#
#    print(word, bins)
#
#print()
#print()
#print()
#print()

#shifted_num = 0
#total_num = 0
#for word in matrix:
##for word in stopwords:
#    if word in stopwords:
#        continue
#    if not word in matrix:
#        continue
#
#    bins = [0,0,0,0,0,0]
#
#    for entry in matrix[word]:
#        if entry > 0.001:
#            bins[0] += 1
#        elif entry > 0.0001:
#            bins[1] += 1
#        elif entry > 0.00001:
#            bins[2] += 1
#        elif entry > 1e-6:
#            bins[3] += 1
#        elif entry > 1e-7:
#            bins[4] += 1
#        else:
#            bins[5] += 1
#        #if entry <= 0.0001:
#        #    bins[0] += 1
#        #elif entry <= 0.001:
#        #    bins[1] += 1
#        #else
#        #    bins[2] += 1
#
#    #if bins[5] > 8000:
#    #    print(word, bins)
#    if bins[0] + bins[1] + bins[3] + bins[4] > 2500:
#        shifted_num += 1
#        print(word, bins)
#    total_num += 1
#
#print(shifted_num)
#print(shifted_num / total_num)
        


for word in stopwords:
    if not word in matrix:
        continue

    #vec_pos = vec_order[word]
    bins = [0,0,0,0,0]

    for word2 in matrix:
        if word in matrix[word2].keys():
            entry = matrix[word2][word]
        else:
            entry = 0

        if entry > 0.001:
            bins[0] += 1
        elif entry > 0.0001:
            bins[1] += 1
        elif entry > 0.00001:
            bins[2] += 1
        elif entry > 1e-6:
            bins[3] += 1
        else:
            bins[4] += 1
        

    print(bins)

print()
print()
print()
    
#random_words = []
#while(len(random_words) < len(stopwords)):
#    word = random.choice(list(matrix.keys()))
#    if not word in stopwords:
#        random_words.append(word)
#random_words = random.sample(list(matrix.keys()), len(stopwords))

exclude_words = []

for i, word in enumerate(matrix.keys()):
    if i % 50000 == 0:
        print(i)
    #vec_pos = vec_order[word]
    bins = [0,0,0,0,0,0]

    for key in matrix:
        if word in matrix[key]:
            entry = matrix[key][word]
        else:
            entry = 0

        if entry > 0.001:
            bins[0] += 1
        elif entry > 0.0001:
            bins[1] += 1
        elif entry > 0.00001:
            bins[2] += 1
        elif entry > 1e-6:
            bins[3] += 1
        elif entry == 0:
            bins[4] += 1
        else:
            bins[5] += 1
        

    if bins[4] < 1e5:
        print(word, bins)
        exclude_words.append(word)


with open("../exclude_words.pkl", "wb") as exclude_file:
    pickle.dump(exclude_words, exlude_file)
    
    

