import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

class WAC_MATT(nn.Module):

	def __init__(self, embedding_dim, vocab_size):
		super(WAC_MATT, self).__init__()
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		embed_init = torch.rand(vocab_size, embedding_dim) ## unif[0,1]
		embed_init = 0.2 * (embed_init - 0.5)
		self.word_embeddings.weight.data = embed_init

		self.u1 = nn.Parameter(torch.randn(embedding_dim)) 
		self.u2 = nn.Parameter(torch.randn(embedding_dim))

		self.linear0 = nn.Linear(embedding_dim,embedding_dim) ## to use in value function
		self.linear1 = nn.Linear(embedding_dim,embedding_dim) 
		self.linear = nn.Linear(embedding_dim,1)
		self.score2prob = nn.Sigmoid()
		self.cosine = nn.CosineSimilarity(dim = 2, eps = 1e-08)


	def forward(self, X, lens):
		batch_size, maxlen = X.size()

		embeds = self.word_embeddings(X)
		
		att1 = self.get_attention(X,lens,self.u1)
		att2 = self.get_attention(X,lens,self.u2)

		#embeds_ave = torch.matmul(att, embeds)
		embeds_ave1 = torch.mul(att1.view(batch_size,-1,1), self.linear0(embeds)).sum(dim = 1)
		embeds_ave2 = torch.mul(att1.view(batch_size,-1,1), self.linear1(embeds)).sum(dim = 1)

		score = self.linear(embeds_ave1 + embeds_ave2)
		prob = self.score2prob(score)
		return prob

	def get_attention(self,X,lens,u):
		batch_size, maxlen = X.size()

		embeds = self.word_embeddings(X)
		## build a mask for paddings
		mask = torch.arange(maxlen)[None,:].float() < lens[:,None].float()
		#print(mask.size())

		## compute attention
		u = torch.unsqueeze(u,0)
		u = torch.unsqueeze(u,0)
		#sim = torch.exp(self.cosine(self.u.weight.data.view(1,1,-1), embeds))
		sim = self.cosine(u, embeds)
		sim = torch.mul(sim.exp(), mask.float())
		
		att = sim/ sim.sum(dim = 1, keepdim=True)
		return att


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

def main():
	from data_pre import data_preprocessing
	from torch import log
	import torch.optim as optim
	import time
	from torch.utils.data import DataLoader, TensorDataset

	torch.manual_seed(1)

	print(torch.cuda.is_available())
	print(torch.__version__)


	'''
	load and prepare data
	'''
	(voc_ix, train_data,test_data, dev_data) = data_preprocessing()
	print("finish preparing data\n")

	'''
	set parameters
	'''
	## set hyperparameters
	VOCAB_SIZE = len(voc_ix) + 1
	EMBEDDING_DIM = 100
	n_epoch = 20
	batch_size = 5000
	eval_per = 20000/batch_size
	PATH = "../model/wac_matt.pt"

	## define model
	model = WAC_MATT(EMBEDDING_DIM, VOCAB_SIZE)
	optimizer = optim.Adagrad(model.parameters(), lr = 1e-2)


	## training
	losses = []
	accs = []
	i = 0
	best_dev_acc = 0

	myloss = torch.nn.BCELoss(weight=None)
	#with torch.autograd.set_detect_anomaly(True):
	start = time.time()
	for epoch in range(n_epoch):
	    print("epoch " + str(epoch))

	    #dataloaders
	    # make sure to SHUFFLE your data
	    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
	    for X,y,lens in train_loader:

	        if i % eval_per == 0:
	            print("time: {}".format(time.time() - start))
	            acc = model.evaluate(dev_data.tensors)
	            if acc > best_dev_acc:
	                best_dev_acc = acc
	                torch.save(model, PATH)
	            print("accuracy on dev: " + str(acc))
	            accs.append(acc)

	        # Step 3. Run our forward pass.
	        prob = model.forward(X, lens)

	        # Step 4. Compute the loss, gradients, and update the parameters by
	        #  calling optimizer.step()
	        #loss_sent = - y*log(prob) - (1-y)*log(1-prob)
	        loss = myloss(prob, y.unsqueeze(1).float())
	        #loss += loss_sent

	        #import pdb; pdb.set_trace()
	        loss.backward()
	        optimizer.step()
	        model.zero_grad()
	        i +=1

	    losses.append(loss.item())
	    runtime = time.time() - start
	print("runtime: " + str(runtime) + "s")

	model_best = torch.load(PATH)
	model_best.eval()
	acc_dev = model_best.evaluate(dev_data.tensors)
	print("best model acc on dev: " + str(acc_dev))
	acc_test = model_best.evaluate(test_data.tensors)
	print("best model acc on test: " + str(acc_test))












		
