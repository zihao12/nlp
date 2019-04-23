import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import log
import torch.optim as optim
import time
from WAC import WAC
from data_pre import data_preprocessing
torch.manual_seed(1)

print(torch.cuda.is_available())
print(torch.__version__)

'''
TO DO: 
* Use sparse embedding and sparse update
* Use batching
    * set grad to be 0
    * set loss to be 0
    * loss.backward() and optimizer
'''

'''
load and prepare data
'''
(voc_ix, trainX, trainy, testX, testy, devX, devy) = data_preprocessing() 
print("finish preparing data\n")

'''
set parameters
'''          
## set hyperparameters 
VOCAB_SIZE = len(voc_ix)
EMBEDDING_DIM = 100
eval_per = 5000
n_epoch = 5
PATH = "../model/wac.pt"

## define model 
model = WAC(EMBEDDING_DIM, VOCAB_SIZE)
optimizer = optim.Adagrad(model.parameters())


'''
training
'''
## training 
losses = []
accs = []
i = 0
best_dev_acc = 0
start = time.time()
for epoch in range(n_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch " + str(epoch))
    total_loss = 0
    for X,y in zip(trainX, trainy):

        if i % eval_per == 0:
            print("time: {}".format(time.time() - start))
            acc = model.evaluate(devX, devy)
            if acc > best_dev_acc:
                best_dev_acc = acc
                torch.save(model, PATH)
            print("accuracy on dev: " + str(acc))
            accs.append(acc)
        # Step 1. clear grad
        model.zero_grad()
        
        # Step 3. Run our forward pass.
        prob = model.forward(X)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = - y*log(prob) - (1-y)*log(1-prob) 
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        i +=1
        
    losses.append(total_loss)
runtime = time.time() - start
print("runtime: " + str(runtime) + "s")

model_best = torch.load(PATH)
model_best.eval()
acc_dev = model_best.evaluate(devX, devy)
print("best model acc on dev: " + str(acc_dev))
acc_test = model_best.evaluate(testX, testy)
print("best model acc on test: " + str(acc_test))


