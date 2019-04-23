## implement Word Averaging Binary Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WAC(nn.Module):

	def __init__(self, embedding_dim, vocab_size):
		super(WAC, self).__init__()
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse = True)
		self.linear = nn.Linear(embedding_dim,1)
		self.score2prob = nn.Sigmoid()

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		embeds_ave = embeds.mean(dim = 0)
		score = self.linear(embeds_ave)
		prob = self.score2prob(score)
		return prob

	def predict(self, sentence):
		prob = self.forward(sentence)
		if prob < 0.5:
			return 0
		else:
			return 1

	def evaluate(self, dataX, datay):
		total = 0
		correct = 0
		for X,y in zip(dataX, datay):
			total += 1
			if self.predict(X) == y:
				correct += 1
		return correct/total
