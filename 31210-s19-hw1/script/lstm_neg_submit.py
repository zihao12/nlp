## submit job for lstm_neg
import os
#rs = [20, 100, 500]
rs = [20]
#fs = [0]
#fs = [round(0.01*x, 3) for x in range(1,11)]
fs = [round(0.01*x, 3) for x in range(1,11)]
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


