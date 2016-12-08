import scipy.io as sio
import sys
import numpy as np
from numpy import linalg as LA

def eig1(A,c,isMax):
    if c>A.shape[0]:
        c = A.shape[0]
    isSym = 1
    if isSym==1:
        eigen_A = LA.eigh(A)
    else:
        eigen_A = LA.eig(A)
    

    v = eigen_A[1]
    d = eigen_A[0]

    if isMax==0:
        eigen_A_sorted = np.sort(d)
        d1 = eigen_A_sorted
        idx = np.argsort(d)
    else:
        eigen_A_sorted = -np.sort(-d)
        d1 = eigen_A_sorted
        idx = np.argsort(-d)
        
    idx1 = idx[0:c]

    eigval = d[idx1]
    eigvec = np.real(v[:,idx1])
    eigval_full = d[idx]

    res = list()
    res.append(eigval)
    res.append(eigvec)
    res.append(eigval_full)

    return res 



    

def L2_distance_1(a,b):
    if a.shape[0]==1:
        a = np.vstack((a,np.repeat(0,a.shape[1])))
        b = np.vstack((b,np.repeat(0,b.shape[1])))
    

    aa = np.matrix(np.sum(np.array(a)*np.array(a),axis = 0))
    
    bb = np.matrix(np.sum(np.array(b)*np.array(b),axis = 0))
    #print(a)
    ab = np.dot(np.matrix(a.T),np.matrix(b))
    
    d1 = np.repeat(0.0,aa.shape[0]*aa.shape[1]*bb.shape[0]*bb.shape[1]).reshape(aa.shape[0]*aa.shape[1],bb.shape[0]*bb.shape[1])
    for i in range(bb.shape[0]*bb.shape[1]):
        d1[:,i] = np.array(aa)[0];

    d2 = np.repeat(0.0,aa.shape[0]*aa.shape[1]*bb.shape[0]*bb.shape[1]).reshape(bb.shape[0]*bb.shape[1],aa.shape[0]*aa.shape[1])
    for i in range(aa.shape[0]*aa.shape[1]):
        d2[i,:] = np.array(bb)[0];

    d = d1+d2 - 2*ab
    d = np.real(d)
    dd= np.repeat(0.0,d.shape[0]*d.shape[1]).reshape(d.shape[0],d.shape[1])
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            dd[i,j] = max(d[i,j],0)

    d = dd
    d_eye = np.repeat(1.0,d.shape[0]*d.shape[1]).reshape(d.shape[0],d.shape[1])
    np.fill_diagonal(d_eye,0)
    d = np.array(d) * np.array(d_eye)

    return d


def umkl(D):
    beta = 1.0 / (D.shape[0]*D.shape[1])
    tol = 1e-4
    u = 20.0
    logU = np.log(u)
    
    #compute Hbeta
    res_hbeta = Hbeta(D,beta)
    
    H = res_hbeta[0]
    thisP = res_hbeta[1]
    
    betamin = -214748365.0
    betamax = 214748365

    Hdiff = H -logU
    
    
    tries = 0.0
    #print(Hdiff)
    while(np.absolute(Hdiff) > tol and tries<30.0):
        #if not, increase or decrease precision
        if Hdiff>0 :
            betamin = beta
            if np.absolute(betamax)==214748365.0:
                np.set_printoptions(precision=3)
                beta = beta * 2.0
            else:
                np.set_printoptions(precision=9)
                beta = (beta + betamax)/2.0
        else:
            np.set_printoptions(precision=9)
            betamax = beta
            if np.absolute(betamin)==-214748365.0:
                beta = beta * 2.0
            else:
                np.set_printoptions(precision=9)
                beta = (beta + betamin)/2.0
        #print(beta)
        #raise ValueError("  ")
    
        np.set_printoptions(precision=9)
        res_hbeta = Hbeta(D,beta)
        
        H = res_hbeta[0]
        thisP = res_hbeta[1]
        
        Hdiff = H-logU
        tries = tries +1.0

    return thisP

def Hbeta(D,beta):
    D = (D - np.min(D))/(np.max(D)- np.min(D) + np.finfo(float).eps)
    #print("D")
    #print(D)
    P = np.exp(-D * beta)
    #print("P")
    #print(P)
    sumP = np.sum(P)
    #print("sumP")
    #print(sumP)
    H = np.log(sumP) + beta * np.sum(np.multiply(D,P)) / sumP
    P=P/sumP
 #   bb=False
#    for i in range(P.shape[1]):
#        P[0,i] = P[0,i]/sumP
#        if np.isnan(P[0,i]):
#            bb= True
#    if bb==True:        
#        for i in range(P.shape[1]):
#            if np.isnan(P[0,i]):
#                P[0,i] = 0
#            else:
#                P[0,i] = 1
        
    
    res=list()
    res.append(H)
    res.append(P)
    return res

if __name__=='__main__':
    data = sio.loadmat('/Users/Zeyin/Desktop/Study/Bioinformatics/SIMLR-SIMLR/data/Test_1_mECS.mat')
    data1 = data["in_X"]
    data1 = data1[0:13,0:13].T
    #print(data1)
    x=np.matrix("0.00015559521874565727, 0.00021246840656371493, 0.00030423172625916806, 0.00046148705046888028, 0.00075122056866200521")
    print (umkl(x))
    














    
    
