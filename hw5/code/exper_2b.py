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
out["pchar"] = pchar
out["acc"] = acc

## save result

out_name = "../output/exper_2b.csv"
# with open(out_name, "w") as csv_file:
#     reader = csv.reader(csv_file)
#     out = dict(reader)

with open(out_name, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in out.items():
       writer.writerow([key, value])

out_name = "../output/exper_2b.pkl"
with open(out_name, "wb") as f:
	pickle.dump(out, f)

pred_name = "../output/exper_2b_B.pkl"
with open(pred_name, "wb") as f:
        pickle.dump(Bs, f)

NBC_name = "../output/exper_2b_NBC.pkl"
with open(NBC_name, "wb") as f:
        pickle.dump(NBC, f)
