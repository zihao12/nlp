## implement nonparametric bayes using gibbs sampling

import numpy as np

## input: list of list
## output: list of list (the final sample)

def data_pre():
	## output is a list of list
	with open("../data/cbt-characters.txt", "r") as f:
		Xs = []
		for x in f.readlines():
			Xs.append(list(x[:-1]))

	with open("../data/cbt-boundaries.txt", "r") as f:
		Bs = []
		for b in f.readlines():
			Bs.append(list(map(int, b[:-1])))

	return Xs, Bs

def initialization(Xs, g, seed = 123):
	np.random.seed(seed)
	Bs = []
	for X in Xs:
		B = []
		L = len(X)
		curr = 1
		while curr < L:
			if np.random.uniform() < g:## place a boundary with prob g
				B.append(1)
			else:
				B.append(0)
			curr += 1
		B.append(1) ## now at the last character
		Bs.append(B)

	## store those boundaries in a dictionary
	Bs_dlist = []
	for B in Bs:
		B_dict = {}
		prev = -1
		L = len(B)
		for idx, b in enumerate(B):
			if b == 1:
				B_dict[idx] = [prev, L-1] ## set the next to be the last index first
				if prev != -1:
					B_dict[prev][1] = idx ## "next" for previous "1"
				prev = idx
		Bs_dlist.append(B_dict)

	## build a counter for segments based on Bs
	seg_counter = {}
	totalN = 0
	for Bdict, X in zip(Bs_dlist, Xs):
		for idx in sorted(Bdict.keys()):
			start = Bdict[idx][0] + 1
			end = idx
			# print(start)
			# print(end)
			seg_name = "".join(X[start: (end + 1)])
			#print(seg_name)
			try:
				seg_counter[seg_name] += 1
			except:
				seg_counter[seg_name] = 1
			totalN += 1
		totalN += 1


	return Bs, Bs_dlist, seg_counter, totalN


def evaluation(Bs, Btruth):
	total = 0
	wrong = 0
	for b,bt in zip(Bs, Btruth):
		total += len(b)
		wrong += sum(abs(np.array(b) - np.array(bt)))
	return (total - wrong) / total

def G0(y, beta, pchar = "uniform"):
	y = list(y)
	out = np.power(1-beta, len(y) - 1) * beta
	if pchar == "uniform":
		out *= 1/55 ## need to count the unique characters
	return out

def gibbs_sampler(Xs, g = 0.5, s = 1, beta = 0.5, niter = 2):
	(Bs, Bs_dlist, seg_counter, totalN) = initialization(Xs, g)
	for n in range(niter):
		#print(n)
		i = 0
		for B, Bdict in zip(Bs, Bs_dlist):
			#print(Bdict)
			#print("# sent: {}".format(i))
			X = Xs[i]
			i += 1
			for idx, b in enumerate(B[:-1]):
				#print("b   = {}".format(b))
				#print("idx = {}".format(idx))
				## now let's sample from P(b | others)
				

				# yprev = "".join(X[Bdict[head][0] + 1: head + 1])
				# ynext = "".join(X[head+1: Bdict[head][1] + 1])
				# yfull = "".join(X[Bdict[head][0] + 1: Bdict[head][1] + 1])
				if b == 1:
					prev = Bdict[idx][0]
					next = Bdict[idx][1]

					yprev = "".join(X[prev+1 :idx+1])
					ynext = "".join(X[idx+1 : next+1])
					yfull = "".join(X[prev+1 : next+1])
					# print("yprev = {}".format(yprev))
					# print("ynext = {}".format(ynext))
					# print("yfull = {}".format(yfull))
					seg_counter[yprev] -= 1
					seg_counter[ynext] -= 1
					N = totalN - 2

				else:
					next = idx + 1
					while B[next] == 0:
						next += 1
					prev = Bdict[next][0]

					yprev = "".join(X[prev+1 : idx+1])
					ynext = "".join(X[idx+1 : next+1])
					yfull = "".join(X[prev+1 : next+1])
					# print("yprev = {}".format(yprev))
					# print("ynext = {}".format(ynext))
					# print("yfull = {}".format(yfull))
					seg_counter[yfull] -= 1
					N = totalN - 1

				

				## sample new b
				try:
					Nyfull = seg_counter[yfull]
				except: 
					Nyfull = 0

				try:
					Nyprev = seg_counter[yprev]
				except:
					Nyprev = 0

				try:
					Nynext = seg_counter[ynext]
				except:
					Nynext = 0

				if yprev == ynext:
					indicator = 1
				else:
					indicator = 0
				## compute probabilities
				p0_ = (Nyfull + s*G0(yfull, beta))/(N + s)
				p1_ = (Nyprev + s*G0(yprev, beta)) * (Nynext + indicator + s*G0(ynext, beta))*(1-g)
				p1_ = p1_/((N + s) * (N + s + 1))
				p0 = p0_/(p0_ + p1_)

				## update B, Bdict and update counts
				if np.random.uniform() < p0:
					B[idx] = 0
					try:
						del Bdict[idx]
					except:
						pass

					try:
						seg_counter[yfull] += 1
					except:
						seg_counter[yfull] = 1

					totalN += 1
				else:
					B[idx] = 1
					Bdict[idx] = [prev, next]

					try:
						seg_counter[yprev] += 1
					except:
						seg_counter[yprev] = 1

					try:
						seg_counter[ynext] += 1
					except:
						seg_counter[ynext] = 1
					totalN += 2

	return Bs






















