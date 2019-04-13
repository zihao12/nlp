import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from LSTMlm_ctx import LSTMlm_ctx
from data_pre import data_preprocessing_ctx
torch.manual_seed(1)

print(torch.cuda.is_available())
print(torch.__version__)


'''
load and prepare data
'''
(voc_ix, data_train, data_test, data_dev) = data_preprocessing_ctx()
print("finish preparing data\n")

'''
set parameters
'''          
## set hyperparameters 
VOCAB_SIZE = len(voc_ix)
EMBEDDING_DIM = 200
HIDDEN_DIM = 200
eval_per = 1000
n_epoch = 2
PATH = "../model/lstmlm_cen_ctx_test.pt"

## define model 
model = LSTMlm_ctx(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# print(len(data_test[0]))
# print(model.forward(data_test[0], data_test[1]).shape)


'''
training
'''
## training 
losses = []
accs = []
best_dev_acc = 0
start = time.time()
for epoch in range(n_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch " + str(epoch))
    total_loss = 0
    for i,Xy in enumerate(data_dev):
        if i % eval_per == 0:
            acc = model.evaluate(data_dev)
            if acc > best_dev_acc:
                best_dev_acc = acc
                torch.save(model, PATH)
            print("accuracy on dev: " + str(acc))
            accs.append(acc)
        # Step 1. clear grad
        model.zero_grad()

        # Step 2. Get inputs ready for the network
        (X,y) = Xy
        
        # Step 3. Run our forward pass.
        probs = model.forward(X, len(y))

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(probs, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses.append(total_loss)
runtime = time.time() - start
print("runtime: " + str(runtime) + "s")

model_best = torch.load(PATH)
model_best.eval()
acc_test = model_best.evaluate(data_test)
print("best model acc on test: " + str(acc_test))
