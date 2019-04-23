import time
import torch

def getdata(name, data_dir="../data/"):
    with open(data_dir + name) as f:
        outX = []
        outy = []
        for sent in f.readlines():
            Xy = sent.split()
            outX.append(Xy[:-1])
            outy.append(int(Xy[-1]))
        outy = torch.Tensor(outy)
    return  outX, outy

def build_dict(trainX, testX, devX):
    import numpy as np
    train_flat = [x for item in trainX  for x in item]
    test_flat = [x for item in testX  for x in item]
    dev_flat = [x for item in devX  for x in item]

    total_unique = np.unique(train_flat + test_flat + dev_flat)
    voc_ix = {}
    for ix, word in enumerate(total_unique):
        voc_ix[word] = ix

    return voc_ix

def corpus2ix(corpus, voc_ix):
    out = []
    for sent in corpus:
        out.append(torch.tensor([voc_ix[w] for w in sent], dtype = torch.long))
    return out
 
def data_preprocessing():
    start = time.time()
    ## get data
    (trainX, trainy) = getdata('senti.train.tsv')
    (testX, testy) = getdata('senti.test.tsv')
    (devX, devy) = getdata('senti.dev.tsv')



    ## build dictionary
    voc_ix = build_dict(trainX, testX, devX)

    ## word2ix
    trainX = corpus2ix(trainX, voc_ix)
    testX = corpus2ix(testX, voc_ix)
    devX = corpus2ix(devX, voc_ix)

    print("data preprocessing takes " + str(time.time() - start))
    return (voc_ix, trainX, trainy, testX, testy, devX, devy)

