## analyze result from exper_2a
import pickle 
import numpy as np
import pandas as pd

ss =[]
betas = []
gs = []
accs = []

for i in range(64):
	outname = "../output/exper_2a_{}.pkl".format(i)
	with open(outname, "rb") as f:
		out = pickle.load(f)
		ss.append(out["s"])
		betas.append(out["beta"])
		gs.append(out["g"])
		accs.append(out["acc"])

out_df = pd.DataFrame(
    {'s': ss,
     'beta': betas,
     'g': gs,
     'acc': accs
    })

out_df.to_csv("../output/exper_2a.csv",index=False)
