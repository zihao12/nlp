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
		mask2 = torch.arange(maxlen)[None,:].float() == lens[:,None].float()-1 ## for taking thought vector
		#mask2.unsqueeze(2)
		## compute attention
		embeds = self.word_embeddings(X) ## [bs, sentlen, embed]

		## Run LSTM to get a thought vector
		tv, _ = self.lstm(embeds.transpose(0,1)) ## need [sentlen, bs, embed]
		tv = tv[mask2.transpose(0,1)].unsqueeze(1)
		tv = tv / tv.sum(2, keepdim = True)
		## get u
		u = torch.unsqueeze(self.u,0)
		u = torch.unsqueeze(u,0) ## now [1,1, embed]
		u = u/u.sum(2, keepdim=True)
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
		




def main():
	from data_pre import data_preprocessing
	from data_pre import data_preprocessing
	from torch import log
	import torch.optim as optim
	import time
	from torch.utils.data import DataLoader, TensorDataset

	(voc_ix, train_data,test_data, dev_data) = data_preprocessing()



	'''
	load and prepare data
	'''
	#(voc_ix, trainX, trainy, testX, testy, devX, devy) = data_preprocessing()
	#print("finish preparing data\n")

	'''
	set parameters
	'''
	## set hyperparameters
	VOCAB_SIZE = len(voc_ix) + 1
	EMBEDDING_DIM = 100
	HIDDEN_DIM = 100
	n_epoch = 20
	batch_size = 5000
	eval_per = 20000/batch_size
	PATH = "../model/wac_attt.pt"

	## define model
	model = WAC_ATT_T(EMBEDDING_DIM,HIDDEN_DIM, VOCAB_SIZE)
	optimizer = optim.Adagrad(model.parameters(), lr = 1e-2)


	## training
	losses = []
	accs = []
	i = 0
	best_dev_acc = 0

	myloss = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
	with torch.autograd.set_detect_anomaly(True):
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
	            loss = myloss(prob, y.float())
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

















