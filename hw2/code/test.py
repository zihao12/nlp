import torch
from WAC_MATT import WAC_MATT
PATH = '../model/wac_matt.pt'
         ## evaluate data
model5 = torch.load(PATH)
print("performance of best model:")
print("model acc on train data: {}".format(model5.evaluate(train_data.tensors))) 
print("model acc on dev data: {}".format(model5.evaluate(dev_data.tensors))) 
print("model acc on test data: {}".format(model5.evaluate(test_data.tensors)))

