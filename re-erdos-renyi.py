import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm


x=np.zeros(81)

y1=np.zeros(81)
ye1=np.zeros(81)
z1=np.zeros(81)
ze1=np.zeros(81)

y2=np.zeros(81)
ye2=np.zeros(81)
z2=np.zeros(81)
ze2=np.zeros(81)

y3=np.zeros(81)
ye3=np.zeros(81)
z3=np.zeros(81)
ze3=np.zeros(81)

y4=np.zeros(81)
ye4=np.zeros(81)
z4=np.zeros(81)
ze4=np.zeros(81)

y5=np.zeros(81)
ye5=np.zeros(81)
z5=np.zeros(81)
ze5=np.zeros(81)

#S=np.zeros(2)
#E=np.zeros(2)
p=[0,0.25,0.5,0.75,1]

for plink in p:
 print(plink)
 
 S=np.zeros(30)
 E=np.zeros(10)

 sg=np.zeros(30)
 gap=np.zeros(10)
 
 i=0
 beta = 4
 
 while (beta >= -4.1):
  print(beta) 
  for k in range(10):
   print(k)
   for j in range(30):
    print(j)
    #For non sparse graph; faster algorithm exists for sparse(small p) graphs
    G = nx.gnp_random_graph(50, plink, directed=False)
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
   
    S[j] = S[j]/(np.log2(200))
    
    ##Calculating Spectral Gap
    v=LA.eigvals(pho)
    sg[j]=np.partition(v, 1)[1] - np.partition(v, 0)[0]
   
   E[k]=np.mean(S)
   gap[k]=np.mean(sg)
  
  x[i] = -beta
  
  if plink == 0:
   y1[i] = np.mean(E) 
   ye1[i]= np.std(E)

   z1[i] = np.mean(gap)
   ze1[i] = np.std(gap)
  
  elif plink == 0.25:
   y2[i] = np.mean(E) 
   ye2[i]= np.std(E)
   z2[i] = np.mean(gap)
   ze2[i] = np.std(gap)
   
  
  elif plink == 0.5:
   y3[i] = np.mean(E) 
   ye3[i]= np.std(E)
   z3[i] = np.mean(gap)
   ze3[i] = np.std(gap)
  
  elif plink == 0.75:
   y4[i] = np.mean(E) 
   ye4[i]= np.std(E)
   z4[i] = np.mean(gap)
   ze4[i] = np.std(gap)
  
  elif plink == 1:
   y5[i] = np.mean(E) 
   ye5[i]= np.std(E)
   z5[i] = np.mean(gap)
   ze5[i] = np.std(gap)
  
  i=i+1
  beta = beta - 0.1

plt.figure(1)
plt.errorbar(x,y1, yerr=ye1, fmt='ro-', label='p=0')
plt.errorbar(x,y2, yerr=ye2, fmt='yo-', label='p=0.25')
plt.errorbar(x,y3, yerr=ye3, fmt='bo-', label='p=0.5')
plt.errorbar(x,y4, yerr=ye4, fmt='go-', label='p=0.75')
plt.errorbar(x,y5, yerr=ye5, fmt='mo-', label='p=1')
plt.xlabel('1/beta(log10 scale)')
plt.ylabel('spectral entropy')
plt.legend(loc='lower right')

plt.figure(2)
plt.yscale('log')
plt.errorbar(x,z1,yerr=ze1, fmt='ro-', label="p=0")
plt.errorbar(x,z2,yerr=ze2, fmt='yo-', label="p=0.25")
plt.errorbar(x,z3,yerr=ze3, fmt='bo-', label="p=0.5")
plt.errorbar(x,z4,yerr=ze4, fmt='go-', label="0.75")
plt.errorbar(x,z5,yerr=ze5, fmt='mo-', label="1")
plt.xlabel('1/beta(log10 scale)')
plt.ylabel('spectral gap')
plt.legend(loc='lower right')
plt.show()
