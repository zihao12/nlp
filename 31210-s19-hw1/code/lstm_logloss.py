import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
torch.manual_seed(1)

print(torch.cuda.is_available())
print(torch.__version__)

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

class LSTMlm(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMlm, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2score = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        scores = self.hidden2score(lstm_out.view(len(sentence), -1))
        probs = F.softmax(scores, dim = 1)
        return probs


def evaluate(model, data):
    total = 0
    correct = 0
    for sentence in data:
        Xy = prepare_sequence(sentence, voc_ix)
        X = Xy[0]
        y = Xy[1]
        total += len(y)
        probs = model(X)
        for idx, pred in enumerate(probs):
            if pred.argmax().item() == y[idx]:
                correct += 1
    return correct/total
        
## set hyperparameters
VOCAB_SIZE = len(voc_ix)
EMBEDDING_DIM = 200
HIDDEN_DIM = 200
eval_per = 1000
PATH = "../model/lstmlm_cen_best.pt"

## define model 
model = LSTMlm(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


## training 
losses = []
accs = []
best_dev_acc = 0
start = time.time()
for epoch in range(20):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch " + str(epoch))
    total_loss = 0
    for i,sentence in enumerate(train):
        if i % eval_per == 0:
            acc = evaluate(model, dev)
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
        probs = model(X)

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
acc_test = evaluate(model_best, test)
print("best model acc on test: " + str(acc_test))
