import numpy as np
from numpy import linalg as LA
import csv

#Add warning signal for input function to be laplacian and numpy.ndarray

#Calculatng the eigen values of the laplacian
def eigs(L):
    eig=LA.eigvals(L)
    return(eig)

#Returns partition function of network
def Z(L,beta):
    Z = np.sum(np.exp(-(beta*eigs(L)))) 
    return(Z)

#Function to calculate spectral entropy for a network given beta
def S(L,beta):
    lambdas = beta*eigs(L)
    H = np.sum( np.exp(-lambdas)*(np.log(Z(L,beta)) + lambdas))/(Z(L,beta)*np.log(2))
    return(H)

#Function to calculate the Helmotz free energy
def F(L,beta):
    Z = Z(L,beta)
    F = (np.log(Z)/np.log(2))/beta
    return(F)

# Returns the avg of a control operator from an eigenvalue spectrum
def H_mean(L, beta):
    lambdas = beta*eigs(L)
    Z_tmp = Z(L,beta)
    val = (np.exp(-lambdas)/Z_tmp) * eigs(L) 
    return(np.sum(val))

# Returns the sq.avg of a control operator from an eigenvalue spectrum
def H_sqmean(L,beta):
    lambdas = beta*eigs(L)
    Z_tmp = Z(L,beta)
    val = (np.exp(-lambdas)/Z_tmp) * (np.square(eigs(L))) 
    return(np.sum(val))

# Returns the heat capacity of a network from an eigenvalue spectrum
# In statistical physics: C = T ∂S/∂T = ∂S/∂log(T) = ... = - ∂S/∂log(beta)

def C(L, beta):
    sq_avg = H_sqmean(L,beta)
    avg = H_mean(L,beta)
    val = (sq_avg - avg**2)*(beta**2)/np.log(2)
    return(val)

def C_resc(L, beta):
    val = C(L, beta)
    val = val*np.log(2)/beta**2
    return(val)

# Returns the efficiency of a network from an eigenvalue spectrum
# Theory: eta = 1 + delta S / delta A
#         delta S = S - log2 N, delta A = -log2 Z + log2 N
#         note that A = -logZ = F*beta; log2 gets factored out in the ratio
#         after algebra: eta = -beta * <H> / log(Z/N)

def eta(L, beta):
    avg = H_mean(L,beta)
    Z_tmp = Z(L,beta)
    N = np.size(eigs(L))
    val = -beta*avg/np.log(Z_tmp/N)
    return(val)

def data_frame(L, beta):
 with open('properties.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["beta", beta])
    writer.writerow(["N", np.size(eigs(L))])
    writer.writerow(["Z", Z(L,beta)])
    writer.writerow(["logZ", np.log2(Z(L,beta))])
    writer.writerow(["S", H(L,beta)])
    writer.writerow(["S.norm", H(L,beta)/np.log2(np.size(eigs(L)))])
    writer.writerow(["F", F(L,beta)])
    writer.writerow(["C", C(L,beta)])
    writer.writerow(["C.resc", C_resc(L,beta)])
    writer.writerow(["eta", eta(L,beta)])
    writer.writerow(["H.mean", H_mean(L,beta)])
    writer.writerow(["H.sqmean", H_sqmean(L,beta)])
    writer.writerow(["lambda.mean", np.mean(eigs(L))])
    writer.writerow(["lambda.sqmean", np.mean(np.square(eigs(L)))])
