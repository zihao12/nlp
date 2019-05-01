import torch

def prepare_sequence(seq, to_ix):
    seq = seq.split()
    X_idxs = [to_ix[w] for w in seq[:-1]]
    y_idxs = [to_ix[w] for w in seq[1:]]
    return (torch.tensor(X_idxs, dtype=torch.long),torch.tensor(y_idxs, dtype=torch.long))

def prepare_data(data, to_ix):
    out = []
    for sent in data:
        out.append(prepare_sequence(sent, to_ix))
    return out

def data_preprocessing():
	with open("../data/bobsue.voc.txt", "r") as f:
	    voc = f.read().splitlines() 
	with open("../data/bobsue.lm.train.txt", "r") as f:
	    train = f.read().splitlines()
	with open("../data/bobsue.lm.test.txt", "r") as f:
	    test = f.read().splitlines()
	with open("../data/bobsue.lm.dev.txt", "r") as f:
	    dev = f.read().splitlines()

	## vocabulary dictionary
	voc_ix = {}
	for idx, val in enumerate(voc):
	    voc_ix[val] = idx

	## prep data
	data_train = prepare_data(train, voc_ix)
	data_test = prepare_data(test, voc_ix)
	data_dev = prepare_data(dev, voc_ix)
	return (voc_ix, data_train, data_test, data_dev)





'''
get Pf for negative sampling
'''
def Pf(voc_ix, data, f):
	voc_count = {}
	for w in voc_ix.items():
	    voc_count[w[1]] = 0
	## count word
	total_word = 0
	for Xy in data:
	    X = Xy[0]
	    for x in X:
	        voc_count[x.item()] += 1
	        total_word += 1
	        
	## turn it into probability
	P = [w[1]/total_word for w in voc_count.items()]
	Pf = [p**f for p in P]
	return Pf



'''

preprocess data with context
'''

def prepare_sequence_ctx(sent, to_ix):
    [ctx,target] =  sent.split("\t")
    y = [to_ix[w] for w in target.split()[1:]]
    X = [to_ix[w] for w in ctx.split()] + [to_ix[w] for w in target.split()[:-1]]
    return (torch.tensor(X, dtype=torch.long),torch.tensor(y, dtype=torch.long))

def prepare_data_ctx(data, to_ix):
    out = []
    for sent in data:
        out.append(prepare_sequence_ctx(sent, to_ix))
    return out

def data_preprocessing_ctx():
    with open("../data/bobsue.voc.txt", "r") as f:
        voc = f.read().splitlines() 
    with open("../data/bobsue.prevsent.train.tsv", "r") as f:
        train = f.read().splitlines()
    with open("../data/bobsue.prevsent.test.tsv", "r") as f:
        test = f.read().splitlines()
    with open("../data/bobsue.prevsent.dev.tsv", "r") as f:
        dev = f.read().splitlines()

    ## vocabulary dictionary
    voc_ix = {}
    for idx, val in enumerate(voc):
        voc_ix[val] = idx

    ## prep data
    data_train = prepare_data_ctx(train, voc_ix)
    data_test = prepare_data_ctx(test, voc_ix)
    data_dev = prepare_data_ctx(dev, voc_ix)
    return (voc_ix, data_train, data_test, data_dev)