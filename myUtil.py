import pickle
def WordCount(file, stopwords='stopwords.txt'):
    wordCnt = {}
    with open(stopwords) as sw:
        swords = [line.strip() for line in sw.readlines()]
    with open(file, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(' ')
            for word in words:
                if word in ',. ': continue
                if word in swords: continue
                wordCnt[word] = wordCnt.get(word, 0) + 1 
   
    return wordCnt

def majorityCnt(wordCnt):
    wordCnt = sorted(wordCnt.items(), key=lambda x: x[1], reverse=True)
    with open('abc.pickle', 'wb') as f:           
        pickle.dump(wordCnt, f, pickle.HIGHEST_PROTOCOL)   
    return wordCnt
