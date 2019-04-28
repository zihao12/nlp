## implement Word Averaging Binary Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WAC_SATT(nn.Module):

	def __init__(self, embedding_dim, vocab_size):
		super(WAC_SATT, self).__init__()
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse = True)
		embed_init = torch.rand(vocab_size, embedding_dim) ## unif[0,1]
		embed_init = 0.2 * (embed_init - 0.5)
		self.word_embeddings.weight.data = embed_init

		self.linear = nn.Linear(embedding_dim,1)
		self.score2prob = nn.Sigmoid()

	def forward(self, X, lens):
		batch_size, maxlen = X.size()
		## build a mask for paddings
		mask = torch.arange(maxlen)[None,:].float() < lens[:,None].float()

		## compute attention
		embeds = self.word_embeddings(X) ## [bs, sentlen, embed]
		sim = torch.mul(embeds, embeds).sum(dim = 2) ## [bs, sentlen]
		sim = torch.mul(sim.exp(), mask.float())
		att = sim/ sim.sum(dim = 1, keepdim=True)
		embeds_ave = torch.mul(att.unsqueeze(2), embeds)
		embeds_ave = embeds_ave.sum(dim = 1)

		## compute score and probability
		score = self.linear(embeds_ave)
		prob = self.score2prob(score)
		return prob

	def predict(self, data):
		X,y,lens = data
		prob = self.forward(X,lens)
		pred = prob > 0.5
		return pred


	def evaluate(self, data):
		pred = self.predict(data)
		_,y,_ = data
		n_correct = (pred.view(-1).float() == y.view(-1).float()).sum().item()
		total = y.size(0)
		return n_correct/total
		



