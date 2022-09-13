from graph_tool.all import *
import graph_tool as gt
import numpy as np
import scipy
import os

edgedirname = "/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Data/snap/treeoflife.interactomes.save"
nodedirname = "/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Data/snap/treeoflife.interactomes.indx"
eigdirname="/mnt/hdd/IISER/5th Year/Domenico/Tree of Life/Code/snap/eig"

if not os.path.exists(eigdirname):
    os.mkdir(eigdirname)

for filename in os.listdir(edgedirname):
    print(filename)
    nodes = np.loadtxt(f'{nodedirname}/{filename}', dtype=str)
    edges = np.loadtxt(f'{edgedirname}/{filename}')

    N=len(nodes[:,0])

    ug1 = Graph(directed=False)
    ug1.add_vertex(N)
    ug1.add_edge_list(edges)

    #Getting the laplacian
    L = gt.spectral.laplacian(ug1) 

    #Calculating Eigen Values
    eig1 = scipy.sparse.linalg.eigs(L, k=N//2, which="LR", return_eigenvectors=False)
    eig2 = scipy.sparse.linalg.eigs(L, k=N-N//2, which="SR", return_eigenvectors=False)
    eig = np.concatenate((eig1, eig2))
    np.savetxt(f'{eigdirname}/{filename}', eig, fmt='%.4f')
