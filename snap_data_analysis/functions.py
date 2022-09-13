import numpy as np
from numpy import linalg as LA
import csv

#Add warning signal for input function to be laplacian and numpy.ndarray

#Calculatng the eigen values of the laplacian
#def eigs(L):
#    eig=LA.eigvals(L)
#    return(eig)

#Returns partition function of network
def Z(eigs,beta):
    Z = np.sum(np.exp(-(beta*eigs))) 
    return(Z)

#Function to calculate spectral entropy for a network given beta
def S(eigs,beta):
    lambdas = beta*eigs
    H = np.sum( np.exp(-lambdas)*(np.log(Z(eigs,beta)) + lambdas))/(Z(eigs,beta)*np.log(2))
    return(H)

#Function to calculate the Helmotz free energy
def F(eigs,beta):
    Z_tmp = Z(eigs,beta)
    F = (np.log(Z_tmp)/np.log(2))/beta
    return(F)

# Returns the avg of a control operator from an eigenvalue spectrum
def H_mean(eigs, beta):
    lambdas = beta*eigs
    Z_tmp = Z(eigs,beta)
    val = (np.exp(-lambdas)/Z_tmp) * eigs
    return(np.sum(val))

# Returns the sq.avg of a control operator from an eigenvalue spectrum
def H_sqmean(eigs,beta):
    lambdas = beta*eigs
    Z_tmp = Z(eigs,beta)
    val = (np.exp(-lambdas)/Z_tmp) * (np.square(eigs)) 
    return(np.sum(val))

# Returns the heat capacity of a network from an eigenvalue spectrum
# In statistical physics: C = T ∂S/∂T = ∂S/∂log(T) = ... = - ∂S/∂log(beta)

def C(eigs, beta):
    sq_avg = H_sqmean(eigs,beta)
    avg = H_mean(eigs,beta)
    val = (sq_avg - avg**2)*(beta**2)/np.log(2)
    return(val)

def C_resc(eigs, beta):
    val = C(eigs, beta)
    val = val*np.log(2)/beta**2
    return(val)

# Returns the efficiency of a network from an eigenvalue spectrum
# Theory: eta = 1 + delta S / delta A
#         delta S = S - log2 N, delta A = -log2 Z + log2 N
#         note that A = -logZ = F*beta; log2 gets factored out in the ratio
#         after algebra: eta = -beta * <H> / log(Z/N)

def eta(eigs, beta):
    avg = H_mean(eigs,beta)
    Z_tmp = Z(eigs,beta)
    N = np.size(eigs)
    val = -beta*avg/np.log(Z_tmp/N)
    return(val)

def data_frame(eigs, beta):
# with open('properties.csv', 'w', newline='') as file:
#    header = ["beta", "N", "Z", "logZ", "S", "S.norm", "F", "C", "C.resc", "eta", " H.mean", "H.sqmean", "lambda.mean", "lambda.sqmean"]
    frame = [beta, np.size(eigs), Z(eigs,beta), np.log2(Z(eigs,beta)), S(eigs,beta), S(eigs,beta)/np.log2(np.size(eigs)), F(eigs,beta), C(eigs,beta), C_resc(eigs,beta), eta(eigs,beta), H_mean(eigs,beta), H_sqmean(eigs,beta), np.mean(eigs), np.mean(np.square(eigs))]

#    writer = csv.writer(file)
#    writer.writerow(header)
#    writer.writerow(data)

    #writer.writerow(["beta", beta])
    #writer.writerow(["N", np.size(eigs)])
    #writer.writerow(["Z", Z(eigs,beta)])
    #writer.writerow(["logZ", np.log2(Z(eigs,beta))])
    #writer.writerow(["S", S(eigs,beta)])
    #writer.writerow(["S.norm", S(eigs,beta)/np.log2(np.size(eigs))])
    #writer.writerow(["F", F(eigs,beta)])
    #writer.writerow(["C", C(eigs,beta)])
    #writer.writerow(["C.resc", C_resc(eigs,beta)])
    #writer.writerow(["eta", eta(eigs,beta)])
    #writer.writerow(["H.mean", H_mean(eigs,beta)])
    #writer.writerow(["H.sqmean", H_sqmean(eigs,beta)])
    #writer.writerow(["lambda.mean", np.mean(eigs)])
    #writer.writerow(["lambda.sqmean", np.mean(np.square(eigs))])
    return(frame)
