## functions for inference
import numpy as np
import itertools

def local_predictor(corpus, em_prob,word2ix, ix2tag):
	em_prob_argmax = np.argmax(em_prob,axis = 0)
	tags_pred = []
	for Xs in corpus:
		tag = []
		for x in Xs:
			pred_ix = em_prob_argmax[word2ix[x]]
			tag.append(ix2tag[pred_ix])
		tags_pred.append(tag)
	return tags_pred
		

def greedy_LR_predictor(corpus, em_prob, trans_prob, word2ix, tag2ix, ix2tag):
	tags_pred =[]
	for Xs in corpus:
		tag = ["<s>"] ## G(0)

		if len(Xs) > 1:
			for x in Xs[:-1]:
				tag_prev = tag[-1]
				logprobs = np.log(trans_prob[tag2ix[tag_prev],:]) + np.log(em_prob[:,word2ix[x]])
				tag.append(ix2tag[np.argmax(logprobs)])

			## for last tag
			x = Xs[-1]
			tag_prev = tag[-1]
			logprobs = np.log(trans_prob[tag2ix[tag_prev],:]) + np.log(em_prob[:,word2ix[x]]) + np.log(trans_prob[:,tag2ix["</s>"]])
			tag.append(ix2tag[np.argmax(logprobs)])
			## get all the tags
			tags_pred.append(tag[1:])
		else:
			x = Xs[0]
			tag_prev = tag[-1]
			logprobs = np.log(trans_prob[tag2ix[tag_prev],:]) + np.log(em_prob[:,word2ix[x]])
			tag.append(ix2tag[np.argmax(logprobs)])
			tags_pred.append(tag[1:])

	return tags_pred

def greedy_RL_predictor(corpus, em_prob, trans_prob, word2ix, tag2ix, ix2tag):
	tags_pred =[]
	for Xs in corpus:
		Xs_rev = Xs[::-1]
		tag = ["</s>"] ## R(T+1)
		for x in Xs_rev[:-1]:
			tag_next = tag[-1]
			logprobs = np.log(trans_prob[:,tag2ix[tag_next]]) + np.log(em_prob[:,word2ix[x]])
			tag.append(ix2tag[np.argmax(logprobs)])
		## for last tag
		x = Xs_rev[-1]
		tag_next = tag[-1]
		logprobs = np.log(trans_prob[:,tag2ix[tag_next]]) + np.log(em_prob[:,word2ix[x]]) + np.log(trans_prob[tag2ix["<s>"],:])
		tag.append(ix2tag[np.argmax(logprobs)])
		## get all the tags
		tag = tag[::-1]
		tags_pred.append(tag[:-1])
	return tags_pred


def Viterbi_predictor(corpus, em_prob, trans_prob, word2ix, tag2ix, ix2tag):
	tags_pred = []

	for Xs in corpus:
		xs = [word2ix[x] for x in Xs]
		(V,L, yT) = Viterbi(xs,em_prob, trans_prob, tag2ix)
		tags = [yT]
		T = len(xs)
		for t in reversed(range(1,T)):
			yprev = L[t, tags[-1]]
			tags.append(yprev)
		tags = tags[::-1]
		tags = [ix2tag[y] for y in tags]
		tags_pred.append(tags)

	return tags_pred

def Viterbi(xs, em_prob, trans_prob, tag2ix):
	T = len(xs)
	M = em_prob.shape[0]
	V = np.empty((0,M))
	L = np.empty((0,M), int)
	## compute V(1,:) & L(1,:)
	x = xs[0]
	v = np.log(trans_prob[tag2ix['<s>'],:]) + np.log(em_prob[:,x])
	l = np.repeat(tag2ix['<s>'],M)
	V = np.concatenate((V, v.reshape(1,-1)), axis = 0)
	L = np.concatenate((L,l.reshape(1,-1)), axis = 0)

	## compute V(t,:) & L(t,:)
	for x in xs[1:]: ## t = 1,...,T-1
		vt_ = np.log(trans_prob) + np.repeat(V[-1,:],M).reshape(-1,M) + np.log(np.repeat(em_prob[:,x],M).reshape(-1,M).T)
		## now vt is M*M; take the column max/argmax to get vt,lt
		vt = vt_.max(axis = 0)
		lt = vt_.argmax(axis = 0) ## to speed up here
		V = np.concatenate((V, vt.reshape(1,-1)), axis = 0)
		L = np.concatenate((L,lt.reshape(1,-1)), axis = 0)

	## get yT
	yT = np.argmax(V[-1,:] + np.log(trans_prob[:, tag2ix['</s>']]))

	return (V,L,yT)

def beam_search_predictor(corpus, em_prob, trans_prob, word2ix, tag2ix, ix2tag, top = 5):
    tags_pred = []

    for Xs in corpus:
        xs = [word2ix[x] for x in Xs]
        B = beam_search(xs,em_prob, trans_prob, tag2ix,top)
        last = B.pop()
        (y,v,i) = last[0]
        tags = [y] ##yT
        while len(B) > 0:
            last = B.pop()
            (y,v,i) = last[i]
            tags.append(y) 
        tags = tags[::-1]
        tags = [ix2tag[y] for y in tags]
        tags_pred.append(tags)

    return tags_pred


def beam_search(xs,em_prob, trans_prob, tag2ix, top):
    B = []
    ## A(1), B(1)
    x = xs[0]
    vs = np.log(em_prob[:,x]) + np.log(trans_prob[tag2ix['<s>'],:])
    item = [(ix,v,-1) for ix,v in enumerate(vs)]
    if len(xs) == 1:
    	B.append(maxb(item,1))
    	return B
    B.append(maxb(item,top))

    ## A(t), B(t)
    for x in xs[1:-1]:
        item = []
        for j,b in enumerate(B[-1]):
            (yh,vh,ih) = b
            vs = np.log(em_prob[:,x]) + np.log(trans_prob[yh,:]) + vh
            As = [(ix,v,j) for ix,v in enumerate(vs)] 
            item.append(As)
        item = list(itertools.chain.from_iterable(item))
        B.append(maxb(item,top))

    ## A(T),B(T)
    x = xs[-1]
    item = []
    for j,b in enumerate(B[-1]):
        (yh,vh,ih) = b
        vs = np.log(em_prob[:,x]) + np.log(trans_prob[yh,:]) + vh + np.log(trans_prob[:,tag2ix['</s>']])
        As = [(ix,v,j) for ix,v in enumerate(vs)]
        item.append(As)
    item = list(itertools.chain.from_iterable(item))
    B.append(maxb(item,1))

    return B

def maxb(item, b):
    item = sorted(item, key=lambda x: x[1])[::-1]
    return item[:b]






























