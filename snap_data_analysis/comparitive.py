from graph_tool.all import *
import graph_tool as gt
import numpy as np
import scipy
import os
#import csv
import functions_beta as fb

eigdirname="/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Code/snap/eig"
comparedirname="/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Code/snap/compare"

if not os.path.exists(comparedirname):
    os.mkdir(comparedirname)

b = np.logspace(-4, 4, num = 81)

for beta in b:
    data = []
    print(beta)
    for filename in os.listdir(eigdirname):
#       print(filename)
        eig = np.loadtxt(f'{eigdirname}/{filename}', dtype=np.complex128)
        eig = np.real(eig)
        data.append([filename] + fb.data_frame(eig, beta))
    np.savetxt(f'{comparedirname}/{beta}.txt', data, fmt='%s')
