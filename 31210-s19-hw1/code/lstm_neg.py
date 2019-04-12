
# coding: utf-8

# In[203]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
torch.manual_seed(1)

## load vocab and turn into idx
with open("../data/bobsue.voc.txt", "r") as f:
    voc = f.read().splitlines() 
voc_ix = {}
for idx, val in enumerate(voc):
    voc_ix[val] = idx

## load training samples
with open("../data/bobsue.lm.train.txt", "r") as f:
    train = f.read().splitlines()
with open("../data/bobsue.lm.test.txt", "r") as f:
    test = f.read().splitlines()
with open("../data/bobsue.lm.dev.txt", "r") as f:
    dev = f.read().splitlines()


## output X, y given a sentence
def prepare_sequence(seq, to_ix):
    seq = seq.split()
    X_idxs = [to_ix[w] for w in seq[:-1]]
    y_idxs = [to_ix[w] for w in seq[1:]]
    return (torch.tensor(X_idxs, dtype=torch.long),torch.tensor(y_idxs, dtype=torch.long))

class LSTMlmNEG(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMlmNEG, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocsize = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings2 = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2score = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        hidden, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        return hidden
    
    
def predict(model, hidden, voc_ix):
    #lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
    voc_embed = model.word_embeddings2(torch.tensor(
        [i for i in range(len(voc_ix))], dtype=torch.long))
    scores = torch.mm(hidden.squeeze(1),torch.t(voc_embed))
    prediction = torch.argmax(scores, 1, keepdim = False)
    return prediction

def evaluate(model, data,voc_ix):
    total = 0
    correct = 0
    for sentence in data:
        Xy = prepare_sequence(sentence, voc_ix)
        X = Xy[0]
        y = Xy[1]
        total += len(y)
        hidden = model.forward(X)
        preds = predict(model, hidden, voc_ix)
        for idx, pred in enumerate(preds):
            if pred == y[idx]:
                correct += 1
    return correct/total
        
def loss_function(hidden, y, model, r):
    ## hidden [sen_len, 1, embed_len]
    ## y [sen_len]
    embed_y = model.word_embeddings2(y)
    posi_score = torch.sum(torch.mul(hidden.squeeze(1),embed_y), dim = 1).sigmoid().log()
    log_posi = torch.sum(posi_score)
    ## [11]
    
    ## sample r words
    idx = torch.empty(len(y),r).random_(0, model.vocsize-1)
    #embed_neg = model.word_embeddings2(idx)
    
    ## get [sen_len, r, embed_len] tensor
    embed_neg = torch.stack([model.word_embeddings2(torch.tensor(ix, dtype=torch.long)) for ix in idx], 0)
    ## sample r words
    idx = torch.empty(len(y),r).random_(0, model.vocsize-1)
    
    ## get [sen_len, r, embed_len] tensor
    embed_neg = torch.stack([model.word_embeddings2(torch.tensor(ix, dtype=torch.long)) for ix in idx], 0)    
    log_sampled = (1/r) * torch.sum((1-torch.sum(torch.mul(hidden, embed_neg), dim = 2).sigmoid()).log())
    
    return -log_posi - log_sampled
    
    


# In[215]:


## set hyperparameters
VOCAB_SIZE = len(voc_ix)
EMBEDDING_DIM = 200
HIDDEN_DIM = 200
eval_per = 1000
r = 30
PATH = "../model/lstmlm_neg_best.pt"

## define model 
model = LSTMlmNEG(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
#loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)


## training 
losses = []
accs = []
best_dev_acc = 0
start = time.time()
for epoch in range(10):  
    print("epoch " + str(epoch))
    total_loss = 0
    for i,sentence in enumerate(train):
        if i % eval_per == 0:
            acc = evaluate(model, dev, voc_ix)
            if acc > best_dev_acc:
                best_dev_acc = acc
                torch.save(model, PATH)
            print("accuracy on dev: " + str(acc))
            accs.append(acc)
        # Step 1. clear grad
        model.zero_grad()

        # Step 2. Get inputs ready for the network
        Xy = prepare_sequence(sentence, voc_ix)
        X = Xy[0]
        y = Xy[1]
        
        # Step 3. Run our forward pass.
        hidden = model.forward(X)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(hidden, y, model, r)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses.append(total_loss)
runtime = time.time() - start
print("runtime: " + str(runtime) + "s")

model_best = torch.load(PATH)
model_best.eval()
acc_test = evaluate(model_best, test,voc_ix)
print("best model acc on test: " + str(acc_test))

