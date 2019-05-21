## implement gibbs sampling
import numpy as np
# implement gibbs sampling for one sentence
# input: sentence (word); K (number of samples); 
# output: samples from posterior 
def gibbs(sent, em_prob, trans_prob, tag2ix, word2ix, K, beta, annealing):
	T = len(sent) + 2 ## including <s> and </s>
	posterior = []
	n_changes = []
	## init
	state = np.random.choice(trans_prob.shape[0], size = T-2, replace=True).tolist()
	state = [tag2ix["<s>"]] + state + [tag2ix["</s>"]]
	posterior.append(state[1:-1])
	## iters of gibbs
	for i in range(K-1):
		beta_ = beta + (i-1)*annealing
		n_change = 0
		for j in range(1,T-1): 
			probs_log = beta_*np.log(trans_prob[state[j-1],:]) + \
			beta_*np.log(trans_prob[:,state[j+1]]) + beta_*np.log(em_prob[:,word2ix[sent[j-1]]])
			probs = np.exp(probs_log)
			probs = probs/probs.sum()

			old = state[j]
			state[j] = np.random.choice(trans_prob.shape[0], size = 1, p = probs)[0]
			if state[j] != old:
				n_change += 1
			posterior.append(state[1:-1])
		n_changes.append(n_change)
	posterior = np.array(posterior)

	return (posterior, n_changes)

def gibbs_predictor(corpus, em_prob, trans_prob, tag2ix, word2ix,ix2tag, K = 10, beta = 1, annealing = 0):
	tags_pred = []
	n_changes_corpus = []
	for sent in corpus:
		(posterior, n_changes) = gibbs(sent, em_prob, trans_prob, tag2ix, word2ix, \
			K, beta = beta,annealing = annealing)
		tags = posterior[-1,:]
		tags = [ix2tag[t] for t in tags]
		tags_pred.append(tags)
		n_changes_corpus.append(n_changes)
	return (tags_pred, n_changes_corpus)





# ## test
# from data_pre import data_preprocessing
# (data_train,data_dev,word2ix, ix2word, tag2ix, ix2tag, em_prob, trans_prob) = data_preprocessing()

# sent = data_dev[0][0]
# K = 100
# out = gibbs(sent, K, em_prob, trans_prob, tag2ix, word2ix)

# print(len(out))
# print(out[:3])
# print(out[-3:])



