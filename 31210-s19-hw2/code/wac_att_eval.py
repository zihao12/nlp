import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import log
import torch.optim as optim
import time
from WAC_ATT import WAC_ATT
from data_pre import data_preprocessing
import numpy as np
import random

torch.manual_seed(1)
random.seed(1)

print(torch.cuda.is_available())
print(torch.__version__)

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
bsize = 500
PATH = "../model/wac_att.pt"

## define model 
model = WAC_ATT(EMBEDDING_DIM, VOCAB_SIZE)
optimizer = optim.Adagrad(model.parameters(), lr = 1e-2, lr_decay = 1e-4)


'''
training
'''
## training 
losses = []
accs = []
i = 0
best_dev_acc = 0
start = time.time()
for epoch in range(n_epoch):  
    print("epoch " + str(epoch))
    total_loss = 0
    loss = 0

    ## shuffle data
    combined = list(zip(trainX, trainy))
    random.shuffle(combined)
    trainX, trainy = zip(*combined)


    for X,y in zip(trainX, trainy):

        if i % eval_per == 0:
            print("time: {}".format(time.time() - start))
            acc = model.evaluate(devX, devy)
            if acc > best_dev_acc:
                best_dev_acc = acc
                torch.save(model, PATH)
            print("accuracy on dev: " + str(acc))
            accs.append(acc)
        
        # Step 3. Run our forward pass.
        prob = model.forward(X)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss_sent = - y*log(prob) - (1-y)*log(1-prob) 
        loss += loss_sent

        if i % bsize == 0:
            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.item()
            loss = 0
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


