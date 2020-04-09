import pandas as pd
import sys
import os
from matplotlib import pyplot as pl
import numpy as np



def plot_output(fname):
    res = pd.read_csv(fname)
    index = [1000*2**x for x in range(8)]
    #res['count'] = index
    for name in res.columns:
        if "_std" in name:
            continue
        std_name = name+"_std"
        mean_row = res[name]
        std_row = res[std_name]
        pl.plot(index, mean_row, label=name)
        pl.fill_between(index, mean_row-std_row, mean_row+std_row, alpha=0.1)

    pl.legend()
    pl.title(fname[fname.index("/")+1:fname.index(".")])
    pl.xlabel('train time (game evals)')
    pl.ylabel('score vs skyline')
    pl.savefig(fname+".png")
    print(res.columns)
    pl.close()

def plot_all_outputs():
    DIRNAME = "outputs"
    for fname in os.listdir(DIRNAME):
        if ".csv" == fname[-4:]:
            path = os.path.join(DIRNAME,fname)
            plot_output(path)
    pass

if __name__ == "__main__":
    #plot_output("outputs/BlottoCombObjective_ncombs7_blottosize10_popsize25.csv")
    plot_all_outputs()
