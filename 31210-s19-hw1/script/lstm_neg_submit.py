## submit job for lstm_neg
import os
rs = [20, 100, 500]
fs = [0,0.1,0.2,0.3,0.4 ,0.5,0.7,1]
path = "./"

for r in rs:
    for f in fs:
        command = "python -u ../code/lstm_neg.py {} {} > ../output/lstm_neg_r{}_f{}.pyout".format(r,f,r,f)
        name = "lstm_neg_r{}_f{}.sbatch".format(r,f)
        with open(path + name, "w") as rsh:
            with open(path + "example.sbatch", "r") as exa:
                for item in exa.readlines():
                    rsh.write(item)
            #rsh.write("\n")
            rsh.write("echo '{}'\n".format(command))
            rsh.write(command)

        ## submit job
        print("sbatch {}".format(name))
        os.system("sbatch {}".format(name))


