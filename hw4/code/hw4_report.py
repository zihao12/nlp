


import sys
#sys.path.insert(0, "../code")
import numpy as np
import itertools
import time
import sys
from gibbs import gibbs, gibbs_predictor, mbr_predictor
from data_pre import data_preprocessing
from misc import compute_prob_log,compute_tag_acc
#import matplotlib.pyplot as plt



print("processing data")
(data_train,data_dev,word2ix, ix2word, tag2ix, ix2tag, em_prob, trans_prob) = data_preprocessing()
em_prob[em_prob == 0] = sys.float_info.min
trans_prob[trans_prob == 0] = sys.float_info.min
(corpus, tags) = data_dev

# print("##############################################################")
# print("                            problem 1.d                       ")
# print("##############################################################")
# Ks = [2,5,10,50,100,500,1000]
# #Ks = [2,5,10]
# for K in Ks:
#     start = time.time()
#     (tags_pred, _) = gibbs_predictor(corpus, em_prob, trans_prob, tag2ix, word2ix,ix2tag, K=K)   
#     runtime = time.time() - start
#     print("Gibbs sampling with K = {}".format(K))
#     print("accuracy : {}".format(compute_tag_acc(tags_pred, tags)))
#     print("log prob : {}".format(compute_prob_log(corpus, tags_pred, trans_prob, em_prob, word2ix, tag2ix)))
#     print("runtime  : {}".format(runtime))


# print("##############################################################")
# print("                            problem 1.e                       ")
# print("##############################################################")
# Ks = [2,5,10,50,100,500,1000]
# betas = [0.5,2,5]
# #Ks = [2,5,10]
# for beta in betas:
#     print("##--------------------------##")
#     print("         beta = {}".format(beta))
#     print("##--------------------------##")
#     for K in Ks:
#         start = time.time()
#         (tags_pred, _) = gibbs_predictor(corpus, em_prob, trans_prob, tag2ix, word2ix,ix2tag, K=K, beta = beta)   
#         runtime = time.time() - start
#         print("Gibbs sampling with K = {}".format(K))
#         print("accuracy : {}".format(compute_tag_acc(tags_pred, tags)))
#         print("log prob : {}".format(compute_prob_log(corpus, tags_pred, trans_prob, em_prob, word2ix, tag2ix)))
#         print("runtime  : {}".format(runtime))


print("##############################################################")
print("                            problem 1.f                       ")
print("##############################################################")

Ks = [2,5,10,50,100,500, 1000]
annealing = 0.1
beta = 0.1
caps = [5,10,20,100]
#Ks = [2,5,10]
for cap in caps:
    print("##--------------------------##")
    print("         beta = {}; cap = {}; annealing = {}".format(beta,cap,annealing))
    print("##--------------------------##")
    for K in Ks:
        start = time.time()
        (tags_pred, _) = gibbs_predictor(corpus, em_prob, trans_prob, tag2ix, word2ix,ix2tag, 
                                         K=K, beta = beta, annealing = annealing, cap = cap)   
        runtime = time.time() - start
        print("Gibbs sampling with K = {}".format(K))
        print("accuracy : {}".format(compute_tag_acc(tags_pred, tags)))
        print("log prob : {}".format(compute_prob_log(corpus, tags_pred, trans_prob, em_prob, word2ix, tag2ix)))
        print("runtime  : {}".format(runtime))


print("##############################################################")
print("                            problem 2.c                       ")
print("##############################################################")

#(corpus, tags) = data_dev
Ks = [2,5,10,50,100,500,1000]
# betas = [0.5,2,5,10]
betas = [1]
annealing = 0

for beta in betas:
    print("##--------------------------##")
    print("         beta = {}".format(beta))
    print("##--------------------------##")
    for K in Ks:
        start = time.time()
        tags_pred = mbr_predictor(corpus, em_prob, trans_prob, tag2ix, word2ix,ix2tag,
                                          K=K, beta = beta, annealing = annealing)
        runtime = time.time() - start
        print("K        : {}".format(K))
        print("accuracy : {}".format(compute_tag_acc(tags_pred, tags)))
        print("log prob : {}".format(compute_prob_log(corpus, tags_pred, trans_prob, em_prob, word2ix, tag2ix)))
        print("runtime  : {}".format(runtime))


    
