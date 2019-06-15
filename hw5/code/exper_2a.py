import sys
from gibbs import data_pre, initialization, evaluation, gibbs_sampler
import time
import numpy as np
import itertools
import pickle


ss = [1,10,100,1000]
betas = [0.01,0.1,0.5,0.9]
gs = [0.01,0.1,0.5,0.9]
combinations = list(itertools.product(ss,betas,gs))
## read the parameter used here
idx = int(sys.argv[1]) - 1
(s,beta, g) = combinations[idx]
print("s = {}; beta = {}; g = {}\n".format(s,beta,g))

(Xs, Btruth) = data_pre()
start = time.time()
Bs = gibbs_sampler(Xs,g = g, s= s, beta = beta, niter=1)
runtime = time.time() - start
print("runtime : {}".format(runtime))

acc = evaluation(Bs, Btruth)

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

