import nltk
import glob


wordlist = []
with open("wordToID_short_all.txt", 'rt', encoding="ISO-8859-1") as f:
    for word in f:
        wordlist.append(word.split(" ")[0])
pos_tags = nltk.pos_tag(wordlist)
posToWords = {}
for pairs in pos_tags:
    if pairs[1] not in posToWords:
        posToWords[pairs[1]] = []
        posToWords[pairs[1]].append(pairs[0])
    else:
        posToWords[pairs[1]].append(pairs[0])
nouns = []
nouns.extend(posToWords["NN"])
nouns.extend(posToWords["NNS"])
print(len(nouns))
nounToID = {}
for i, noun in enumerate(nouns):
    nounToID[noun] = i
with open('nounToID_abstract', 'a') as f2:
    for key, value in nounToID.items():
        f2.write(str(key)+' '+str(value)+'\n')
txt_files_clean = glob.glob("nipstxt/nips12/short_*.txt")
for file in sorted(txt_files_clean):
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        doc_wordID_short = ''
        for line in f:
            doc_words = line.split()
            for word in doc_words:
                if word in nounToID:
                    doc_wordID_short += str(nounToID[word])
                    doc_wordID_short += ' '
        with open('nipstoyall/'+'abs_nID'+file[-8:-4]+'.txt', 'a') as f2:
            f2.write(doc_wordID_short)
            f2.close()
