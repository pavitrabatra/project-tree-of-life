from graph_tool.all import *
import graph_tool as gt
import numpy as np
import scipy
import os
#import csv
import functions_beta as fb

eigdirname="/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Code/snap/eig"
analysisdirname="/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Code/snap/analysis"

if not os.path.exists(analysisdirname):
    os.mkdir(analysisdirname)

b = np.logspace(-4, 4, num = 81)

for filename in os.listdir(eigdirname):
    print(filename)
    eig = np.loadtxt(f'{eigdirname}/{filename}', dtype=np.complex128)
    eig = np.real(eig)
    data=[]
    for beta in b:
       print(beta)
       data.append(fb.data_frame(eig,beta))
    np.savetxt(f'{analysisdirname}/{filename}', data, fmt='%.4f')
