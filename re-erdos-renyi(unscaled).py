import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

#For non sparse graph; faster algorithm exists for sparse(small p) graphs
G = nx.gnp_random_graph(200, 0.5, seed=None, directed=False)
#nx.draw(G, with_labels=True)

beta = 1/10000
x=np.empty(9)
y=np.empty(9)

while (beta <= 10000):

    L = nx.laplacian_matrix(G)
    #The above generates a scipy sparse array and not a matrix(Better for comutational purposes?);

    #Converting the sparse array to matrix 
    L = L.todense()

    pho_tmp = expm(-beta*L)
    Z = pho_tmp.trace()
    pho = expm(-beta*L)/Z
    Lp = np.matmul(L,pho)
    Tr = Lp.trace()

    S = np.log2(Z) + (beta/(np.log(2)) * Tr)
    print(S/(np.log2(200)))
   
    x = np.append(x, 1/beta)
    y = np.append(y, S/(np.log2(200)))

    beta = beta *10

print(x)
print(y)
plt.plot(x,y)
plt.show()
