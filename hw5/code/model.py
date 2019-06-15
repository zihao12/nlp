## implement algorithms
import numpy as np
import pdb
import time
import collections
from collections import Counter


def data_pre(textfile = "../data/cbt-characters.txt"):
	## output is a list of list
	with open(textfile, "r") as f:
		Xs = []
		for x in f.readlines():
			Xs.append(list(x[:-1]))

	with open("../data/cbt-boundaries.txt", "r") as f:
		Bs = []
		for b in f.readlines():
			Bs.append(list(map(int, b[:-1])))

	return Xs, Bs

def get_char_probs(X_list):
    x_cont = np.concatenate(X_list)
    C = Counter(x_cont)
    prob_unif = 1/len(C)
    
    prob_unigram = collections.defaultdict(float)
    total_freq = sum([C[x] for x in C])
    for c in C:
        prob_unigram[c] = C[c]/total_freq

    return prob_unif, prob_unigram


def evaluation(Bs, Btruth):
	total = 0
	wrong = 0
	for b,bt in zip(Bs, Btruth):
		total += len(b)
		wrong += sum(abs(np.array(b) - np.array(bt)))
	return (total - wrong) / total

def accuracy(pred, true):
    total_true, total_count = 0, 0
    for p, t in zip(pred, true):
        p, t = np.array(p)[:-1], np.array(t)[:-1]
        total_count += p.size
        total_true += np.sum(p == t)
    return total_true/total_count


# def init(X_list, g, seed = 123):
# 	np.random.seed(seed)
# 	B_list = []
# 	for X in X_list:
# 		B = []
# 		L = len(X)
# 		curr = 1
# 		while curr < L:
# 			if np.random.uniform() < g:## place a boundary with prob g
# 				B.append(1)
# 			else:
# 				B.append(0)
# 			curr += 1
# 		B.append(1) ## now at the last character
# 		B_list.append(B)
# 	return B_list

def init(X_list, g, seed = 123):
	np.random.seed(seed)
	def init_one(X,g):
		L = len(X)
		B = np.random.choice(2, L, [1-g, g])
		B[-1] = 1
		return B

	B_list = [init_one(X,g).tolist() for X in X_list]
	return B_list

## pchar : (prob_type, probs)
def G0(y, beta, pchar):
	y = list(y)
	out = np.power(1-beta, len(y) - 1) * beta
	prob_type = pchar[0]
	probs = pchar[1]

	if prob_type == "uniform":
		out *= np.exp(len(y)*np.log(probs)) 
	if prob_type == "unigram":
		probs_log = sum([np.log(probs[c]) for c in y])
		out *= np.exp(probs_log)
	return out

def new_seg(seg_counter, y, beta,pchar):
	seg = "".join(y)
	if seg not in seg_counter.keys():
		myG0 = G0(y, beta,pchar)
		seg_counter[seg] = [0, myG0]
	return seg_counter

def set_seg_count(seg_counter, y, count):
	seg = "".join(y)
	seg_counter[seg][0] = count
	return seg_counter

def get_count(seg_counter,y):
	seg = "".join(y)
	return seg_counter[seg][0]

def get_G0(seg_counter,y):
	seg = "".join(y)
	return seg_counter[seg][1]



def inference(X_list,Btruth, config, bpa = True):
	s = config["s"]
	beta = config["beta"]
	g = config["g"]
	niter = config["niter"]
	pchar = config["pchar"]

	## schedule is (g_start, rate, cap)
	try:
		schedule = config["schedule"]
	except:
		schedule = (1,0,1)

	alpha_, rate, cap = schedule ## we raise probability to its alpha-power

	print("get char probability")
	prob_uniform, prob_unigram = get_char_probs(X_list)
	if pchar == "uniform":
		pchar = (pchar, prob_uniform)
	if pchar == "unigram":
		pchar = (pchar, prob_unigram)
	## initialization
	## TODO:
	print("initialization")
	B_list = init(X_list, g)
	## Build dictionary for storing segments
	print("build dictionary")
	seg_counter = {}
	total_n = 0
	x_cont = np.concatenate(X_list)
	b_cont = np.concatenate(B_list)

	split_idxs = np.nonzero(b_cont)[0] + 1
	segs = np.split(x_cont, split_idxs)
	
	#pdb.set_trace()
	for seg in segs:
		y = seg
		seg = "".join(seg)
		if seg not in seg_counter.keys():
			seg_counter = new_seg(seg_counter, y, beta, pchar)## TODO: add base_prob here
		seg_counter[seg][0] += 1
		total_n += 1
	print("get {} unique segments\n".format(len(seg_counter)))
	print("total {} segments\n".format(total_n))

	total_NBC = []
	## gibbs sampling
	for i in range(niter):
		alpha = min(alpha_ + i*rate, cap)
		print("--------- niter {} -------------\n".format(i+1))
		start = time.time()
		NBC = 0
		for j in range(len(X_list)):
			X, B = X_list[j], B_list[j]
			L = len(X)
			probs = np.random.rand(L-1)
			next_idx = (np.nonzero(B)[0] + 1).tolist()[::-1]
			prev = 0

			for k in range(L-1): ## k starts from 1 to L-1
				if k+1 == next_idx[-1]:
					next_idx.pop()
				# TODO: FIX BUG
				#print(k)
				next = next_idx[-1]
				yfull = X[prev : next]
				yprev = X[prev : k+1]
				ynext = X[k+1 : next]

				seg_counter = new_seg(seg_counter, yfull, beta, pchar)
				seg_counter = new_seg(seg_counter, yprev, beta, pchar)
				seg_counter = new_seg(seg_counter, ynext, beta, pchar)

				nyfull = get_count(seg_counter, yfull)
				nyprev = get_count(seg_counter, yprev)
				nynext = get_count(seg_counter, ynext)
				n_other = total_n

				
				if B[k] == 0:
					nyfull -= 1
					n_other -= 1
				else:
					nyprev -= 1
					nynext -= 1
					n_other -= 2

				if yprev == ynext:
					indicator = 1
				else:
					indicator = 0

				p0_ = (nyfull + s*get_G0(seg_counter,yfull))/(n_other + s)
				p1_ = (nyprev + s*get_G0(seg_counter,yprev)) * (nynext + indicator + s*get_G0(seg_counter,ynext))*(1-g)
				p1_ = p1_/((n_other + s) * (n_other + s + 1))

				p0_ = np.power(p0_,alpha)
				p1_ = np.power(p1_,alpha)
				p0 = p0_/(p0_ + p1_)

				new = 0 if probs[k] < p0 else 1
					
				if B[k] != new:
					NBC += 1
				B[k] = new

				if new == 0:
					n_other +=  1
					nyfull += 1
				else: 
					n_other += 2
					nyprev += 1
					nynext += 1
					prev = k + 1

				seg_counter = set_seg_count(seg_counter, yfull, nyfull)
				seg_counter = set_seg_count(seg_counter, yprev, nyprev)
				seg_counter = set_seg_count(seg_counter, ynext, nynext)
				total_n = n_other

		
		runtime = time.time() - start
		print("  time {}\n".format(runtime))
		if bpa:
			acc = accuracy(B_list, Btruth)
			print("   acc {}\n".format(acc))
		print("   NBC {}\n".format(NBC))
		total_NBC.append(NBC)

	return B_list, total_NBC


















