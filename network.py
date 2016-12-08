import scipy.io as sio
import numpy as np
import math
from numpy import linalg as LA
def networkDiffusion(A,K):
    np.fill_diagonal(A,0)

    sign_A = A
    sign_A = sign_A>0
    sign_A = (sign_A -1) + sign_A

    P = np.array(dominateSet((np.absolute(A)),min(K,A.shape[0]-1))) * np.array(sign_A)
    
    DD = np.sum(np.absolute(P),axis=1)
    
    np.fill_diagonal(P,DD+1)
    #print("P is ")
    

    P = transitionFields(P)
    #print(P)
    eigen_P = LA.eig(P)
    
    U = eigen_P[1]
    D = eigen_P[0]
    #print(D)
    d=(D + np.finfo(float).eps).real
    
    
    alpha = 0.8
    beta = 2
    #print(1.0 - alpha*(np.array(d)**beta))
    d = ((1.0-alpha)*d) / (1.0 - alpha*(np.array(d)**beta))
    
    
    l = np.matrix(d).shape[0] * np.matrix(d).shape[1]
    D = np.repeat(0.0,l*l).reshape(l,l)
    np.fill_diagonal(D,(d.real))
    #print(D)
    W= np.dot(np.dot(np.matrix(U),np.matrix(D)),np.matrix(U.T))
    
    diagonal_matrix = np.matrix(np.repeat(0.0,W.shape[0]*W.shape[1]).reshape(W.shape[0],W.shape[1]))
    
    np.fill_diagonal(diagonal_matrix,1)
    
    divider = np.repeat(0.0,W.shape[0]*W.shape[1]).reshape(W.shape[0],W.shape[1])
    for i in range(W.shape[1]):
        divider[:,i] = 1-np.diag(W)
    
    W = (np.array(W) * np.array((1.0 - diagonal_matrix)))/ divider
    #print(divider)
    di = np.diag(D)
    np.fill_diagonal(D,di[::-1])
    W = np.dot(np.matrix(np.diag(DD)),np.matrix(W))
    W = (W + W.T) / 2.0

    W[W<0] = 0

    return W

##input is a matrix
def dominateSet(affMatrix, NR_OF_KNN):
    
    PNN_matrix = np.repeat(0.0,affMatrix.shape[0]*affMatrix.shape[1]).reshape(affMatrix.shape[0],affMatrix.shape[1])    
    res_sort = np.sort(-(affMatrix),axis=1)
    res_sort = -res_sort    
    res_sort_indices = np.argsort(-(affMatrix),axis=1)    
    res = res_sort[:,0:NR_OF_KNN]
    inds = np.repeat(0,affMatrix.shape[0]*NR_OF_KNN).reshape(affMatrix.shape[0],NR_OF_KNN)
    for i in range(NR_OF_KNN):
        inds[:,i] = np.arange(affMatrix.shape[0])
    #print(inds)
    loc = res_sort_indices[:,0:NR_OF_KNN]
    #print(loc)
    #print(asvectorCol((np.matrix(loc))))
    indices = (asvectorCol(np.matrix(loc))-1)[0] * (affMatrix.shape[0]) +  (asvectorCol(inds))[0]
    #print(indices)
    PNN = (asvectorCol(PNN_matrix))
    #print(PNN)
    resvector = (asvectorCol(res))
    #print(resvector)
    k=0
    row = asvectorCol(np.matrix(loc))[0]
    #print(row)
    col =  (asvectorCol(inds))
    #print(col)
    #print(resvector)
    for i in range(len(row)):
        PNN_matrix[row[i],col[i]] = resvector[k]
        k=k+1
    #print(PNN_matrix)         
    PNN_matrix = PNN_matrix.T
    #print(PNN_matrix) 
    PNN_matrix = (PNN_matrix + PNN_matrix.T)/2.0
    #print(PNN_matrix) 
    return np.matrix(PNN_matrix)

def asvectorCol(w):
    v = w[:,0]
    for i in range(w.shape[1]-1):
        v = np.concatenate((v,w[:,i+1]),axis=0)
    return np.array(v.T)

def asvectorRow(w):
    v=w[0,:]
    for i in range(w.shape[0]-1):
        v = np.concatenate((v,w[i+1,:]),axis=0)
    return v

    

def transitionFields(W):
    zero_index = np.where(np.sum(W,axis=1)==0)[0]
    #print("zero_index is ")
    #print(zero_index)
    W = dn(W,"ave")

    w = np.sqrt(np.sum(np.absolute(W),axis=0)+np.finfo(float).eps)
    
    divider = np.repeat(0.0,W.shape[0]*W.shape[1]).reshape(W.shape[0],W.shape[1])
    for i in range(divider.shape[1]):
        divider[:,i] = w
    divider = np.matrix(divider).T
    W = W / divider
    
    W = np.dot(W,W.T)
    
    W[zero_index,:] = 0
    W[:,zero_index] = 0

    return W

def dn(w,type):
    D = np.sum(w,axis = 0)
    if type=="ave":
        D = np.matrix(1.0/D)
        lengthD = D.shape[0] * D.shape[1]
        D_temp = np.repeat(0.0,lengthD*lengthD).reshape(lengthD,lengthD)
        k=0
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                D_temp[k,k] = D[i,j]
                k=k+1
        D = D_temp
        wn = np.dot(D,w)

    elif type=="gph":
        D = 1.0/ np.sqrt(D)
        lengthD = D.shape[0] * D.shape[1]
        D_temp = np.repeat(0.0,lengthD*lengthD).reshape(lengthD,lengthD)
        k=0
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                D_temp[k,k] = D[i,j]
                k=k+1

        D = D_temp
        wn = np.dot(np.matrix(D),np.dot(np.matrix(w),np.matrix(D)))

    return wn


if __name__=='__main__':
    data = sio.loadmat('/Users/Zeyin/Desktop/Study/Bioinformatics/SIMLR-SIMLR/data/Test_1_mECS.mat')
    data1 = data["in_X"]
    data1 = data1[0:20,0:20].T
    print(dn(data1,"ave"))
   
