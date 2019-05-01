from data_pre import data_preprocessing
from data_pre import data_preprocessing
from WAC_ATT_T import WAC_ATT_T
import torch
import torch.nn as nn
import torch.nn.functional as F
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
