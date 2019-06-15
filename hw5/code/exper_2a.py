import sys
from model import data_pre, inference, accuracy
import time
import numpy as np
import itertools
import pickle

niter = 1
pchar = "uniform"


ss = [0.1,0.5,5,20]
betas = [0.01,0.1,0.5,0.9]
gs = [0.01,0.1,0.5,0.9]
combinations = list(itertools.product(ss,betas,gs))
## read the parameter used here
idx = int(sys.argv[1]) - 1
(s,beta, g) = combinations[idx]

config = {"s":s, "beta":beta,"g":g, "niter":niter, "pchar":pchar}
print("s = {}; beta = {}; g = {}\n".format(s,beta,g))

(Xs, Btruth) = data_pre()
start = time.time()
Bs,NBC = inference(Xs,Btruth,config)
runtime = time.time() - start
print("runtime : {}".format(runtime))

# import pdb
# pdb.set_trace()
acc = accuracy(Bs, Btruth)

print("acc : {}".format(acc))

out = {}
out["s"] = s
out["beta"] = beta
out["g"] = g
out["acc"] = acc

## save result
out_name = "../output/exper_2a_{}.pkl".format(idx)
with open(out_name, "wb") as f:
	pickle.dump(out, f)

pred_name = "../output/exper_2a_{}_B.pkl".format(idx)
with open(pred_name, "wb") as f:
        pickle.dump(Bs, f)

NBC_name = "../output/exper_2a_{}_NBC.pkl".format(idx)
with open(NBC_name, "wb") as f:
        pickle.dump(NBC, f)
