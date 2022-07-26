import numpy as np
import networkx as nx
from scipy.linalg import logm, expm

#Add warning signal for input function to be laplacian and numpy.ndarray

#Function to calculate density matrix for network given beta
def d_matrix(L,beta):
    rho_tmp = expm(-beta*L)
    Z = rho_tmp.trace()
    rho = expm(-beta*L)/Z
    return(rho)

#Function to calculate spectral entropy for a network given beta
def spectral_entropy(L,beta):
    rho = d_matrix(L,beta)
    s_tmp = rho @ logm(rho)/np.log(2)
    s = -s_tmp.trace()
    s_norm = s/(np.log2(200))
    return(s_norm)

#Function to calculate kullback leibler divergence of network 1(g1) wrt network 2(g2)
def kl_div(L1,beta_1,L2,beta_2):
    rho_1 = d_matrix(L1,beta_1) 
    rho_2 = d_matrix(L2,beta_2)

    kldiv_tmp = rho_1 @ (logm(rho_1)/np.log(2)-logm(rho_2)/np.log(2))
    kldiv= kldiv_tmp.trace()
    return(kldiv)

def js_div(L1,beta_1,L2,beta_2):
    rho_1 = d_matrix(L1,beta_1) 
    rho_2 = d_matrix(L2,beta_2)
    rho_3 = (rho_1 + rho_2)/2
    s_tmp = rho_3 @ logm(rho_3)/np.log(2)
    s = -s_tmp.trace()
    s_norm = s/(np.log2(200))

    jsdiv = s_norm -0.5*spectral_entropy(L1,beta_1)-0.5*spectral_entropy(L2,beta_2)
    return(jsdiv)
