## helper function for loading result
import os
import torch
import random
def show_result(name, train_data, dev_data,test_data,sample = False):
	print("show result for {}".format(name))
	random.seed(1)
	k = 1000
	modeldir = "../model"
	PATH = os.path.join(modeldir, name)
	model = torch.load(PATH)
	print("performance of best model:")
	if sample:
		X,y,lens = train_data.tensors
		idx = random.sample(range(X.size(0)),k)
		train_data.tensors = (X[idx,:], y[idx], lens[idx])

	print("model acc on train data: {}".format(model.evaluate(train_data.tensors))) 
	print("model acc on dev data: {}".format(model.evaluate(dev_data.tensors))) 
	print("model acc on test data: {}".format(model.evaluate(test_data.tensors)))
	return(model)

