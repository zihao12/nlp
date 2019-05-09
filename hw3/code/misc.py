## some helper functions
import numpy as np


def compute_prob_log(corpus, tags, trans_prob, em_prob, word2ix, tag2ix):
	'''
	My way of computing log prob for each sentence is as follow:
	x:  _ , | x1,x2,..., xT
	y: <s>, | y1,y2,..., yT
	yh: y1, | y2,..,yT, </s>
	'''
	log_prob_total = 0
	for Xs,ys in zip(corpus, tags):
	    log_prob = 0
	    yhs = ys[1:] + ['</s>']
	    for x, y, yh in zip(Xs, ys, yhs):
	        log_prob += np.log(trans_prob[tag2ix[y],tag2ix[yh]])
	        log_prob += np.log(em_prob[tag2ix[y], word2ix[x]])
	    log_prob += np.log(trans_prob[tag2ix['<s>'], tag2ix[ys[0]]])
	    log_prob_total += log_prob

	return log_prob_total 

def compute_tag_acc(tags_pred,tags_true):
	total = 0
	correct = 0
	for y_preds, y_true in zip(tags_pred, tags_true):
		total += len(y_preds)
		correct += sum([1 if y == yt else 0 for y,yt in zip(y_preds, y_true)])
		#correct += np.sum(np.equal(y_preds,y_true,dtype=object))

	return correct/total