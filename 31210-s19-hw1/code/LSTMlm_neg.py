import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class LSTMlm_neg(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMlm_neg, self).__init__()
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
    
    
    def predict(self, hidden, voc_ix):
        voc_embed = self.word_embeddings2(torch.tensor(
            [i for i in range(len(voc_ix))], dtype=torch.long))
        scores = torch.mm(hidden.squeeze(1),torch.t(voc_embed))
        prediction = torch.argmax(scores, 1, keepdim = False)
        return prediction

    def evaluate(self, data,voc_ix):
        total = 0
        correct = 0
        for Xy in data:
            (X,y) = Xy
            total += len(y)
            hidden = self.forward(X)
            yp = self.predict(hidden, voc_ix)
            for g, p in zip(y,yp):
                if g == p:
                    correct += 1
        return correct/total
            
    def loss_function(self, hidden, y, r, Pf):
        ## hidden [sen_len, 1, embed_len]
        ## y [sen_len]
        embed_y = self.word_embeddings2(y)
        posi_score = torch.sum(torch.mul(hidden.squeeze(1),embed_y), dim = 1).sigmoid().log()
        log_posi = torch.sum(posi_score)
        ## [11]
        
        ## sample r words
        idx = torch.tensor(random.choices(list(range(self.vocsize)), Pf, k = r*len(y))).view(len(y), r)
        ## get [sen_len, r, embed_len] tensor
        embed_neg = torch.stack([self.word_embeddings2(torch.tensor(ix, dtype=torch.long)) for ix in idx], 0)    
        log_sampled = (1/r) * torch.sum((1-torch.sum(torch.mul(hidden, embed_neg), dim = 2).sigmoid()).log())
        
        return -log_posi - log_sampled

    def loss_function_chinge(self, hidden, y, r, Pf):
        ## hidden [sen_len, 1, embed_len]
        ## y [sen_len]
        embed_y = self.word_embeddings2(y)
        posi_score = torch.sum(torch.mul(hidden.squeeze(1),embed_y), dim = 1)
        
        ## sample r words
        idx = torch.tensor(random.choices(list(range(self.vocsize)), Pf, k = r*len(y))).view(len(y), r)
        ## get [sen_len, r, embed_len] tensor
        embed_neg = torch.stack([self.word_embeddings2(torch.tensor(ix, dtype=torch.long)) for ix in idx], 0) 
        neg_score = torch.sum(torch.mul(hidden, embed_neg), dim = 2) ## [sen_len, r]
        scorem = 1 - posi_score[:,None] + neg_score
        scorem[scorem < 0] = 0
        
        return torch.sum(scorem)














        
