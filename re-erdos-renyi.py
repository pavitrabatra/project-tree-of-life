import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


beta = 4
x=np.zeros(81)
y=np.zeros(81)
ye=np.zeros(81)
i=0
S=np.zeros(100)
E=np.zeros(10)


while (beta >= -4.1):
 for k in range(10):
  print(k)
  for j in range(100):
   #For non sparse graph; faster algorithm exists for sparse(small p) graphs
   G = nx.gnp_random_graph(200, 0.5, directed=False)
   #nx.draw(G, with_labels=True)
   L = nx.laplacian_matrix(G)
   #The above generates a scipy sparse array and not a matrix(Better for comutational purposes?);
  
   #Converting the sparse array to matrix 
   L = L.todense()
   pho_tmp = expm(-10**(beta)*L)
   Z = pho_tmp.trace()
   pho = expm(-10**(beta)*L)/Z
   Lp = np.matmul(L,pho)
   Tr = Lp.trace()

   S[j] = np.log2(Z) + (10**(beta)/(np.log(2)) * Tr)
   print(j)
   
   S[j] = S[j]/(np.log2(200))
  E[k]=np.mean(S)
  print(E)


 y[i] = np.mean(E)
 print(y)
 ye[i]= np.std(E)
 print(ye)
 x[i] = -beta
 i=i+1
 beta = beta - 0.1

print(x)
print(y)
print(ye)
plt.errorbar(x,y,yerr=ye, fmt='ro-')
plt.show()
