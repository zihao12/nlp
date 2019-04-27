import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def getdata(name, data_dir="../data/"):
    with open(data_dir + name) as f:
        outX = []
        outy = []
        for sent in f.readlines():
            Xy = sent.split()
            outX.append(Xy[:-1])
            outy.append(int(Xy[-1]))
        #outy = torch.Tensor(outy)
    return  outX, np.array(outy)

def build_dict(trainX, testX, devX):
    import numpy as np
    train_flat = [x for item in trainX  for x in item]
    test_flat = [x for item in testX  for x in item]
    dev_flat = [x for item in devX  for x in item]

    total_unique = np.unique(train_flat + test_flat + dev_flat)
    voc_ix = {}
    for ix, word in enumerate(total_unique):
        voc_ix[word] = ix + 1

    return voc_ix

def corpus2ix(corpus, voc_ix):
    out = []
    for sent in corpus:
        out.append([voc_ix[w] for w in sent])
    return out

def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    end_ix = np.zeros(len(reviews_int), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = review + zeroes
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
        end_ix[i] = review_len
    
    return features, end_ix


def data_preprocessing():
    start = time.time()
    (trainX, trainy) = getdata('senti.train.tsv')
    (testX, testy) = getdata('senti.test.tsv')
    (devX, devy) = getdata('senti.dev.tsv')

    ## build dictionary
    voc_ix = build_dict(trainX, testX, devX)

    ## word2ix
    trainX = corpus2ix(trainX, voc_ix)
    testX = corpus2ix(testX, voc_ix)
    devX = corpus2ix(devX, voc_ix)

    ## padding
    Seq_length = max([len(x) for x in trainX] + \
                     [len(x) for x in testX]+[len(x) for x in devX])
    trainX,  train_endi = pad_features(trainX, Seq_length)
    testX,  test_endi = pad_features(testX, Seq_length)
    devX,  dev_endi = pad_features(devX, Seq_length)

    train_data = TensorDataset(torch.from_numpy(trainX),\
                               torch.from_numpy(trainy),torch.from_numpy(train_endi))
    test_data = TensorDataset(torch.from_numpy(testX),\
                               torch.from_numpy(testy),torch.from_numpy(test_endi))
    dev_data = TensorDataset(torch.from_numpy(devX),\
                               torch.from_numpy(devy),torch.from_numpy(dev_endi))

    print("runtime: {}".format(time.time() - start))
    return (voc_ix, train_data,test_data, dev_data)






