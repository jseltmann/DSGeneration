from matrix_class import DS_matrix

matrix = DS_matrix("../50k_stopwords/_matrix.pkl")

sent_filename = "../generated_sentences.txt"

with open(sent_filename, "w") as sent_file:
    for i in range(100000):
        if i % 5000 == 0:
            print(i)
        sent = matrix.generate_bigram_sentence()
        sent_str = " ".join(sent) + "\n"
        sent_file.write(sent_str)

