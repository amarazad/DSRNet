import matplotlib.pyplot as plt
import seaborn as sns

import os, glob
def readfile(dir, file):
    with open(os.path.join(dir, file), 'r') as f:
         values = f.readlines()
    return [float(v.split(',')[0].split('(')[1]) for v in values]

def gplot(y, path, y1 = []):
    plt.plot(y, 'b')
    plt.plot(y1, 'r')
    plt.xlabel('No. of steps')
    plt.ylabel('Loss values')
    #plt.show()
    plt.savefig(path)

import argparse
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--dir", default='output', type=str,help="Directory where the output is stored")
arg = parser.parse_args()

tr_loss = readfile(arg.dir, 'train_results.txt')
eval_loss = readfile(arg.dir, 'eval_results.txt')
print(len(tr_loss), len(eval_loss))
print(os.path.join(arg.dir, "train_results.txt"))
gplot(tr_loss[-100:], os.path.join(arg.dir, 'loss.png'), eval_loss[-100:])
