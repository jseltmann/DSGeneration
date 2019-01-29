
with open("SICK.txt") as sick_file:
  lines = sick_file.readlines()

sents = set()
for line in lines[1:]:
  sent = line.split('\t')[1]
  sents.add(sent)
  if len(sents) == 100:
    break

with open("combined_sents.txt", "w") as outfile:
  for sent in sents:
    outfile.write(sent)
    outfile.write("\n")

with open("DSGeneration/brown_sents/2000_freq_sents_from_brown.txt") as brown_file:
  sents = set()
  for line in brown_file:
    sents.add(line)
    if len(sents) == 100:
      break

with open("combined_sents.txt", "a") as outfile:
  for sent in sents:
    outfile.write(sent)

with open("DSGeneration/brown_sents/2000_freq_long_sents.txt") as brown_file:
  sents = set()
  for line in brown_file:
    sents.add(line)
    if len(sents) == 100:
      break

with open("combined_sents.txt", "a") as outfile:
  for sent in sents:
    outfile.write(sent)
    
