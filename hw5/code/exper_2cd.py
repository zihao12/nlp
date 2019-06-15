import sys
from model import data_pre, inference, accuracy
import time
import numpy as np
import itertools
import pickle

niter = 200
pchar = "unigram"
s = 5
beta = 0.1
g = 0.001

## different combinations of schedule
starts = [0.1]
rates = [0.005,0.01,0.03,0.05,0.07,0.1,0.2]
caps = [1,3,5,10]
combinations = list(itertools.product(starts,rates,caps))
idx = int(sys.argv[1]) - 1
schedule = combinations[idx]

config = {"s":s, "beta":beta,"g":g, "niter":niter, "pchar":pchar, "schedule":schedule}
print(config)

(Xs, Btruth) = data_pre()
start = time.time()
Bs,NBC = inference(Xs,Btruth,config)
runtime = time.time() - start
print("runtime : {}".format(runtime))

# import pdb
# pdb.set_trace()
acc = accuracy(Bs, Btruth)

print("acc : {}".format(acc))

out = config
out["acc"] = acc


## save result
out_name = "../output/exper_2cd_{}.pkl".format(idx)
with open(out_name, "wb") as f:
	pickle.dump(out, f)

pred_name = "../output/exper_2cd_{}_B.pkl".format(idx)
with open(pred_name, "wb") as f:
        pickle.dump(Bs, f)

NBC_name = "../output/exper_2cd_{}_NBC.pkl".format(idx)
with open(NBC_name, "wb") as f:
        pickle.dump(NBC, f)
