## implement Word Averaging Binary Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

class WAC_ATT(nn.Module):

	def __init__(self, embedding_dim, vocab_size):
		super(WAC_ATT, self).__init__()
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		embed_init = torch.rand(vocab_size, embedding_dim) ## unif[0,1]
		embed_init = 0.2 * (embed_init - 0.5)
		self.word_embeddings.weight.data = embed_init

		self.u = nn.Parameter(torch.randn(embedding_dim)) ## good to use embedding? ## nn.Parameter


		self.linear = nn.Linear(embedding_dim,1)
		self.score2prob = nn.Sigmoid()
		self.cosine = nn.CosineSimilarity(dim = 2, eps = 1e-08)


	def forward(self, X, lens):
		batch_size, maxlen = X.size()

		embeds = self.word_embeddings(X)
		## build a mask for paddings
		mask = torch.arange(maxlen)[None,:] < lens[:,None]
		#print(mask.size())

		## compute attention
		u = torch.unsqueeze(self.u,0)
		u = torch.unsqueeze(u,0)
		#sim = torch.exp(self.cosine(self.u.weight.data.view(1,1,-1), embeds))
		sim = self.cosine(u, embeds)
		sim = torch.mul(sim.exp(), mask.float())
		
		att = sim/ sim.sum(dim = 1, keepdim=True)
		#embeds_ave = torch.matmul(att, embeds)
		embeds_ave = torch.mul(att.view(batch_size,-1,1), embeds).sum(dim = 1)
		score = self.linear(embeds_ave)
		prob = self.score2prob(score)
		return prob

	def predict(self, X,l):
		prob = self.forward(X.unsqueeze(0),l)
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
