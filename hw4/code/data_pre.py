## data preprocessing
import numpy as np
import itertools


def get_data(datadir, dataname):
	## get list of sentences; list of corresponding tags
	corpus = []
	tags = []
	with open(datadir + dataname, "r") as f:
	    end = False
	    sent = []
	    tag = []
	    for xy in f.readlines():
	        if xy == '\n':
	            end = True
	            corpus.append(sent)
	            tags.append(tag)
	            sent = []
	            tag = []
	        else:
	            x,y = xy.split()
	            sent.append(x)
	            tag.append(y)
	return (corpus, tags)


def data_preprocessing():
	dataname_train = "en_ewt.train"
	dataname_dev = "en_ewt.dev"
	datadir = "../data/"

	## get list of sentences; list of corresponding tags
	data_train = get_data(datadir, dataname_train)
	data_dev = get_data(datadir, dataname_dev)
	(corpus, tags) = data_train

	## get voc_words, voc_tag
	voc_word = set(list(itertools.chain.from_iterable(corpus)))
	voc_tags = set(list(itertools.chain.from_iterable(tags)))
	voc_tags.add('<s>')
	voc_tags.add('</s>')

	## word2ix, ix2word
	word2ix = {}
	ix2word = {}
	for i, w in enumerate(voc_word):
	    word2ix[w] = i
	    ix2word[i] = w

	## tag2ix, ix2tag
	tag2ix = {}
	ix2tag = {}
	for i, t in enumerate(voc_tags):
	    tag2ix[t] = i
	    ix2tag[i] = t

	trans_count = np.zeros((len(voc_tags), len(voc_tags)))
	## trans_prob[i,j] is #(tagi before tagj) bigram pair
	for tag in tags:
	    ys = ['<s>'] + tag
	    yhs = tag + ['</s>']
	    for y,yh in zip(ys,yhs):
	        trans_count[tag2ix[y], tag2ix[yh]] += 1

	        
	em_count = np.zeros((len(voc_tags), len(voc_word)))
	## em_prob[i,j] is # (wordj | tagi)
	for sent, tag in zip(corpus, tags):
	    for x,y in zip(sent, tag):
	        em_count[tag2ix[y],word2ix[x]] += 1

	        
	## get trans_prob
	trans_lam = 0.1
	trans_prob = trans_count + trans_lam
	trans_prob[:,tag2ix['<s>']] = 0
	trans_prob[tag2ix['<s>'],tag2ix['</s>']] = 0
	trans_prob = trans_prob/trans_prob.sum(axis = 1).reshape(-1,1)
	trans_prob[tag2ix['</s>'],:] = 0 ## does not affect scaling

	# get em_prob
	em_lam = 0.001
	em_prob = em_count + em_lam
	em_prob = em_prob/em_prob.sum(axis = 1).reshape(-1,1)
	em_prob[tag2ix['<s>'] ,:] = 0
	em_prob[tag2ix['</s>'] ,:] = 0

	return (data_train, data_dev, word2ix, ix2word, tag2ix, ix2tag, em_prob, trans_prob)





