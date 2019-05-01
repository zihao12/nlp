import operator
import pickle
import matplotlib.pyplot as plt

def error_analysis(model ,data):
    err_pair = []
    for Xy in data:
        (X,y) = Xy
        yhat = model.predict(X)
        for yg, yp in zip(y, yhat):
            if yg != yp:
                err_pair.append((yg.item(), yp))
    return err_pair

def common_error_pair(model, data, top, ix_voc):
    err_pair = error_analysis(model,  data)
    err_pair_count = {}
    for p in err_pair:
        if p not in err_pair_count.keys():
            err_pair_count[p] = 1
        else:
            err_pair_count[p] += 1            
    err_pair_count_sorted = sorted(err_pair_count.items(), key=operator.itemgetter(1), reverse=True)
    err_pair_frequent = err_pair_count_sorted[:top]
    # print("most common error pair\n")
    # for ((yg,yp),c ) in err_pair_frequent:
    #     print("({} {}): {}".format(ix_voc[yg], ix_voc[yp], c))
    return err_pair_frequent

def error_analysis_ctx(model ,data):
    err_pair = []
    for Xy in data:
        (X,y) = Xy
        lastlen = len(y)
        yhat = model.predict(X,lastlen)
        for yg, yp in zip(y, yhat):
            if yg != yp:
                err_pair.append((yg.item(), yp))
    return err_pair

def common_error_pair_ctx(model, data, top, ix_voc):
    err_pair = error_analysis_ctx(model, data)
    err_pair_count = {}
    for p in err_pair:
        if p not in err_pair_count.keys():
            err_pair_count[p] = 1
        else:
            err_pair_count[p] += 1
            
    err_pair_count_sorted = sorted(err_pair_count.items(), key=operator.itemgetter(1), reverse=True)
    err_pair_frequent = err_pair_count_sorted[:top]
    #print("most common error pair\n")
    # for ((yg,yp),c ) in err_pair_frequent[:top]:
    #     print("({} {}): {}".format(ix_voc[yg], ix_voc[yp], c))
    return err_pair_frequent

def plot_acc_neg(f,r):
    name = "../output/lstmlm_neg_f{}_r{}_acc.csv".format(f,r)
    with open(name, "rb") as a:
        acc = pickle.load(a)
    plt.plot(acc)
    plt.xlabel("n_sentences (thousands)")
    plt.ylabel("accuarcy rate on dev")
    plt.title("n_sentence VS accuaracy with f={} and r={}".format(f,r))
    plt.show()

def plot_acc_negchinge(f,r):
    name = "../output/lstmlm_neg_f{}_r{}_acc.csv".format(f,r)
    with open(name, "rb") as a:
        acc = pickle.load(a)
    plt.plot(acc)
    plt.xlabel("n_sentences (thousands)")
    plt.ylabel("accuarcy rate on dev")
    plt.title("n_sentence VS accuaracy with f={} and r={}".format(f,r))
    plt.show()
