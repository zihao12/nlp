import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from LSTMlm_neg import LSTMlm_neg
from data_pre import data_preprocessing
from data_pre import Pf
torch.manual_seed(1)

print(torch.cuda.is_available())
print(torch.__version__)

'''
load and prepare data
'''
(voc_ix, data_train, data_test, data_dev) = data_preprocessing()
print("finish preparing data\n")

'''
set parameters
'''          
## set hyperparameters
VOCAB_SIZE = len(voc_ix)
EMBEDDING_DIM = 200
HIDDEN_DIM = 200
eval_per = 1000
n_epoch = 20

r = int(sys.argv[1])
f = float(sys.argv[2])
PATH = "../model/lstmlm_neg_f{}_r{}.pt".format(f,r)
print(PATH)

## define model
model = LSTMlm_neg(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
#loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 5*1e-4, eps = 1e-3)


Pf = Pf(voc_ix, data_train, f)
losses = []
accs = []
best_dev_acc = 0
start = time.time()
for epoch in range(n_epoch):  
    print("epoch " + str(epoch))
    total_loss = 0
    for i,Xy in enumerate(data_train):
        if i % eval_per == 0:
            curr_runtime = time.time() - start
            curr_sent = len(data_train) * epoch + i
            acc = model.evaluate(data_dev, voc_ix)
            if acc > best_dev_acc:
                best_dev_acc = acc
                torch.save(model, PATH)
                best_runtime = curr_runtime
                best_sent = curr_sent
            print("accuracy on dev: {}; nsentence: {}; runtime: {}".format(acc, curr_sent, curr_runtime))
            accs.append(acc)
        # Step 1. clear grad
        model.zero_grad()

        # Step 2. Get inputs ready for the network
        (X,y) = Xy
        
        # Step 3. Run our forward pass.
        hidden = model.forward(X)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = model.loss_function(hidden, y, r, Pf)
        total_loss += loss.item()

        loss.backward()
        ## clip the norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

    losses.append(total_loss)
    
runtime = time.time() - start
print("tolal runtime: " + str(runtime) + "s")
print("runtime until max acc: {}".format(best_runtime))
print("n_sent until max acc : {}".format(best_sent))
print("n_sent per second    : {}".format(best_sent/best_runtime))
model_best = torch.load(PATH)
model_best.eval()
acc_dev = model_best.evaluate(data_dev, voc_ix)
print("best model acc on test: " + str(acc_dev))
acc_test = model_best.evaluate(data_test, voc_ix)
print("best model acc on test: " + str(acc_test))

