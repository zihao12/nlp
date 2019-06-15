


from gibbs import data_pre, initialization, evaluation, gibbs_sampler
import time
import numpy as np




(Xs, Btruth) = data_pre()




# with open("../data/cbt-characters.txt", "r") as f:
#     dic = []
#     for x in f.readlines()[:-1]:
#         dic += list(x) 
# len(set(dic))


start = time.time()
Bs = gibbs_sampler(Xs, niter=10)
runtime = time.time() - start
print("runtime : {}".format(runtime))

acc = evaluation(Bs, Btruth)
print("acc : {}".format(acc))

