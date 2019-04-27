## implement Word Averaging Binary Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class WAC(nn.Module):

	def __init__(self, embedding_dim, vocab_size):
		super(WAC, self).__init__()
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse = True)
		embed_init = torch.rand(vocab_size, embedding_dim) ## unif[0,1]
		embed_init = 0.2 * (embed_init - 0.5)
		self.word_embeddings.weight.data = embed_init


		self.linear = nn.Linear(embedding_dim,1)
		self.score2prob = nn.Sigmoid()

	def forward(self,X,lens):
		embeds = self.word_embeddings(X)
		## build a mask for paddings
		maxlen = X.size(1)
		mask = torch.arange(maxlen)[None,:] < lens[:,None]
		## set padding to be 0
		embeds[~mask] = float(0)
		print(mask.size())
		## average for non-padding embedding
		embeds_ave = (embeds.mean(dim = 1).t()*(maxlen/lens.float())).t()
		score = self.linear(embeds_ave)
		prob = self.score2prob(score)
		return prob

	def predict(self, X,l):
		prob = self.forward(X.unsqueeze(0), l)
		if prob < 0.5:
			return 0
		else:
			return 1

	def evaluate(self, data):
		(dataX, datay,lens) = data
		total = 0
		correct = 0
		for X,y,l in zip(dataX, datay, lens):
			total += 1
			if self.predict(X,l=torch.tensor([l], dtype = torch.long)) == y:
				correct += 1
		return correct/total
