import numpy as np
import networkx as nx
from scipy.linalg import logm, expm

#Check possibility for optimization in defintion of spectral entropy and density matrix (returning 2 values for only spectral entropy function and not defining density matrix seperately)

#Function to calculate density matrix for network given beta
def d_matrix(g,beta):
    L = nx.laplacian_matrix(g)
    #The above generates a scipy sparse array and not a matrix; Converting the sparse array to matrix 
    L = L.toarray()
    rho_tmp = expm(-beta*L)
    Z = rho_tmp.trace()
    rho = expm(-beta*L)/Z
    return(rho)

#Function to calculate spectral entropy for network given beta
def spectral_entropy(g,beta):
    L = nx.laplacian_matrix(g)
    #The above generates a scipy sparse array and not a matrix; Converting the sparse array to matrix 
    L = L.toarray()
    rho_tmp = expm(-beta*L)
    Z = rho_tmp.trace()
    rho = expm(-beta*L)/Z
    s_tmp = L @ rho
    Tr = s_tmp.trace()
    s = np.log2(Z) + (beta/(np.log(2)) * Tr)
    s_norm = s/(np.log2(200))
    return(s_norm)

#Function to calculate kullback leibler divergence of network 1(g1) wrt network 2(g2)
def kl_div(g1,beta_1,g2,beta_2):
    rho_1 = d_matrix(g1,beta_1) 
    rho_2 = d_matrix(g2,beta_2)

    kldiv_tmp = rho_1 @ (logm(rho_1)/np.log(2)-logm(rho_2)/np.log(2))
    kldiv= kldiv_tmp.trace()
    return(kldiv)

def js_div(g1,beta_1,g2,beta_2):
    rho_1 = d_matrix(g1,beta_1) 
    rho_2 = d_matrix(g2,beta_2)
    rho_3 = (rho_1 + rho_2)/2
    s_tmp = rho_3 @ logm(rho_3)/np.log(2)
    s = -s_tmp.trace()
    s_norm = s/(np.log2(200))

    jsdiv = s_norm -0.5*spectral_entropy(g1,beta_1)-0.5*spectral_entropy(g2,beta_2)
    return(jsdiv)
