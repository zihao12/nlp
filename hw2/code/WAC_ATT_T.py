## implement Word Averaging Binary Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WAC_ATT_T(nn.Module):

	def __init__(self, embedding_dim, hidden_dim,vocab_size, lam=0.1):
		super(WAC_ATT_T, self).__init__()
		self.lam = lam
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		embed_init = torch.rand(vocab_size, embedding_dim) ## unif[0,1]
		embed_init = 0.2 * (embed_init - 0.5)
		self.word_embeddings.weight.data = embed_init

		self.lstm = nn.LSTM(embedding_dim, hidden_dim)
		self.u = nn.Parameter(torch.randn(embedding_dim)) 
		self.linear = nn.Linear(embedding_dim,1)
		self.score2prob = nn.Sigmoid()
		self.cosine = nn.CosineSimilarity(dim = 2, eps = 1e-08)


	def forward(self, X, lens):
		lam = self.lam
		batch_size, maxlen = X.size()
		## build a mask for paddings
		mask = torch.arange(maxlen)[None,:].float() < lens[:,None].float()

		## compute attention
		embeds = self.word_embeddings(X) ## [bs, sentlen, embed]

		## Run LSTM to get a thought vector
		tv, _ = self.lstm(embeds.transpose(0,1)) ## need [sentlen, bs, embed]
		tv = tv[:,-1,:] ## now  [bs, 1, embed]
		#tv = tv / tv.sum(2, keepdim = True)
		## get u
		u = torch.unsqueeze(self.u,0)
		u = torch.unsqueeze(u,0) ## now [1,1, embed]
		#u = u/u.sum(2, keepdim=True)
		u = (1-lam)*u + lam*tv

		sim = self.cosine(u, embeds)
		sim = torch.mul(sim.exp(), mask.float())
		
		att = sim/ sim.sum(dim = 1, keepdim=True)
		#embeds_ave = torch.matmul(att, embeds)
		embeds_ave = torch.mul(att.view(batch_size,-1,1), embeds).sum(dim = 1)
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
		



