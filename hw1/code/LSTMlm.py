import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        #probs = F.softmax(scores, dim = 1)
        return scores

    def predict(self, X):
        yhat = []
        scores = self.forward(X)
        for idx, pred in enumerate(scores):
            yhat.append(pred.argmax().item())
        return yhat

    def evaluate(self, data):
        total = 0
        correct = 0
        for Xy in data:
            (X,y) = Xy
            yhat = self.predict(X)
            total += len(y)
            for yp,yt in zip(yhat, y):
                if yp == yt:
                    correct += 1
        return correct/total    