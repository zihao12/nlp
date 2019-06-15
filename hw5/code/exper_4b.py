import sys
from model import data_pre, inference, accuracy
import time
import numpy as np
import itertools
import pickle
import csv

niter = 100
pchar = "unigram"
s = 5
beta = 0.1
g = 0.001

config = {"s":s, "beta":beta,"g":0.001, "niter":niter, "pchar":pchar}
print("s = {}; beta = {}; g = {}\n".format(s,beta,g))

(Xs, Btruth) = data_pre(textfile = "../data/bobsue.lm.train.txt")
start = time.time()
Bs,NBC = inference(Xs,Btruth,config, bpa = False)
runtime = time.time() - start
print("runtime : {}".format(runtime))

## save result
pred_name = "../output/exper_4a_B.pkl"
with open(pred_name, "wb") as f:
        pickle.dump(Bs, f)

