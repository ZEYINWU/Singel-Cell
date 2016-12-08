import numpy as np
import network as nw
import multikernel as mk
from numpy import linalg as LA
import scipy.io as sio
import util_silmr as us
import tsne as ts
from scipy.cluster.vq import vq, kmeans, whiten
import project as pj
 
def SIMLR(X,c,no_dim ,k=10, if_impute = False,normalize = False,cores_ratio=5):
    #print(X)
    if if_impute==True :
        X = X.T
        X_zeros = np.where(X==0)
        if len(X_zeros)>0:
            R_zeros = X_zeros[0]
            C_zeros = X_zeros[1]
            ind = (C_zeros) * X.shape[0] + R_zeros
            mm = np.array(np.mean(X,axis=0))[0]
            m = mm
            for j in range(X.shape[0]-1):
                mm = np.concatenate((mm,m),axis=0)
            xx = asvectorRow(X)           
            for i in ind:
                xx[i] = mm[i]
            X = xx.reshape(X.shape[0],X.shape[1])
        X = X.T

    if normalize == True :
        X = X.T
        X = X - np.min(X)
        X = X / np.max(X)
        C_mean = (np.mean(X,axis=0))
        X = (X-C_mean).T

    NITER = 40
    num = X.shape[1]
    r=-1
    beta = 0.5

    D_Kernels = mk.multiKernel(X.T,cores_ratio)
    
    alphaK = 1.0 / np.repeat(len(D_Kernels),len(D_Kernels))

    l=D_Kernels[0].shape[0]
    ll=D_Kernels[0].shape[1]
    distX = np.repeat(0.0,l*ll).reshape(l,ll)
    for z in range(len(D_Kernels)):
        distX = distX + D_Kernels[z]
    
    
    distX = distX / len(D_Kernels)
    
    #print(len(D_Kernels))
    
    res = np.sort(distX)
    
    indices = np.argsort(distX)
    
    distX1 = np.repeat(0.0,distX.shape[0]*distX.shape[1]).reshape(distX.shape[0],distX.shape[1])
    idx = np.repeat(0.0,distX.shape[0]*distX.shape[1]).reshape(distX.shape[0],distX.shape[1])

    distX1 = res
    idx = indices

    A = np.repeat(0.0,num*num).reshape(num,num)
    di = distX1[:,1:(k+2)]
    
    rr = 0.5 * (k*di[:,k] - np.sum(di[:,range(k)],axis=1))
    
    id = idx[:,1:(k+2)]
    
    numerator = np.repeat(0.0, di.shape[0]*di.shape[1]).reshape(di.shape[0],di.shape[1])
    for i in range(di.shape[1]):
        numerator[:,i] = di[:,k]
    numerator = numerator - di
    
    temp = np.matrix(k*di[:,k] - np.sum(di[:,0:k],axis=1)) + np.finfo(float).eps
    
  #  temp=np.array(temp[0,:])
    
    denominator = np.repeat(0.0,(temp.shape[1]) * di.shape[1]).reshape((temp.shape[1]),di.shape[1])
    
    for i in range(temp.shape[1]):
        denominator[i,:] = temp[:,i]
    
    temp = numerator / denominator
    
    a = np.repeat(0.0, num * di.shape[1]).reshape(num,di.shape[1])
    for i in range(di.shape[1]):
        a[:,i] = np.arange(num)
    
    row = asvectorCol(np.matrix(a))[0]
    
    col = asvectorCol(np.matrix(id))[0]
    tempvector = asvectorCol(np.matrix(temp))[0]

    k=0
    for i in range(len(row)):
        A[row[i],col[i]] = tempvector[k]
        k=k+1
    
    
    if r<=0:
        r = np.mean(np.array(rr))

    lambda1=max(np.mean(np.array(rr)),0)
    #print(lambda1)

    #print("A is ")
    #print(A)
    A[np.isnan(A)] = 0
    A0 = (A+A.T)/2.0
    S0 =  -(distX-np.max(distX))
    
    #print(distX)
    
    S0 = nw.networkDiffusion(S0,k)
    
    #print(S0)
    S0 = nw.dn(S0,"ave")
    
    S = S0
    d0 = (np.sum(S,axis = 0))
    l = d0.shape[1]
    D0 = np.repeat(0.0 , l*l).reshape(l,l)
    np.fill_diagonal(D0,d0)
    
    L0 = D0-S
    
    eig1_res = us.eig1(L0,c,0)
    F_eig1 = eig1_res[1]
    temp_eig1 = eig1_res[0]
    #print(temp_eig1)
    #raise ValueError("stop")
    evs_eig1 =eig1_res[2]
    #print(evs_eig1)
    #raise ValueError("stop")
    converge = list()
    for iter in range(NITER):
        distf = us.L2_distance_1((F_eig1.T),(F_eig1.T))
        #print(distf)
        #raise ValueError("stop")
        A = np.repeat(0.0,num*num).reshape(num,num)
        b = idx[:,1:idx.shape[1]]
        
        a = np.repeat(0,num*b.shape[1]).reshape(num,b.shape[1])
        for i in range(b.shape[1]):
            a[:,i] = np.arange(num)
        
        inda =np.matrix(np.concatenate((np.matrix(nw.asvectorCol(a)),np.matrix(nw.asvectorCol(b))),axis=0)).T
        
        
        add=list()
        
        for z in range(inda.shape[0]):
            np.set_printoptions(precision=9)
            add.append(((distX[inda[z,0],inda[z,1]]) + lambda1 * distf[inda[z,0],inda[z,1]]) / 2.0 /r)
                   
        ad = np.matrix(add).reshape(num,b.shape[1]).T 

        c_input = -np.matrix(ad).T
        c_output = np.matrix(ad).T
        

        ad = pj.projsplx(c_input,c_output)        
        ad1 = asvectorCol(ad)[0]

        A1 = asvectorCol(A)
        
        for i in range(inda.shape[0]):
            A[inda[i,0],inda[i,1]] = ad1[i]
        
        
        A[np.isnan(A)] = 0
        A = (A + A.T)/2.0
        S = (1-beta)* S + beta * A
        
        #print(S)
        S = nw.networkDiffusion(np.array(S),k)
        
        
        D = np.repeat(0.0,S.shape[1]*S.shape[1]).reshape(S.shape[1],S.shape[1])
        np.fill_diagonal(D,np.sum(S,axis=0))
        
        L = D - S
        
        F_old = F_eig1
        F_eig1 = eig1_res[1]
        temp_eig1 = eig1_res[0]
        
        ev_eig1 = eig1_res[2]
        
        evs_eig1 = np.concatenate((np.matrix(evs_eig1),np.matrix(ev_eig1)),axis=0)
#        print(evs_eig1)
#        raise ValueError("stop")
        DD = list()
        for i in range(len(D_Kernels)):
            temp = np.array(np.finfo(np.float32).eps + D_Kernels[i]) * np.array(S+np.finfo(np.float32).eps)
            DD.append(np.mean(np.sum(temp,axis = 0)))
                  
        alphaK0 = us.umkl(np.matrix(DD))
         
        alphaK0 = alphaK0 / np.sum(alphaK0)
        
        alphaK = (1-beta)*alphaK + beta * alphaK0
        alphaK = alphaK / np.sum(alphaK)
        
        fn1 = np.sum(ev_eig1[0:c])
        fn2 = np.sum(ev_eig1[0:c+1])
        
        converge.append(fn2-fn1)
        #print(converge)
        #raise ValueError("stop")
        if iter < 10:
            if ev_eig1[len(ev_eig1)-1] > 0.00001 :
                lambda1 = 1.5 * lambda1
                r = r /1.01
        else:
            if np.matrix(converge)[0,iter-1]>np.matrix(converge)[0,iter-2]:
                S = S_old
                if np.matrix(converge)[0,iter-2] > 0.2:
                    raise Warning("Maybe you should set a larger value of c")
                break
        S_old = S
        
        #compute Kbeta
        #print(alphaK)
        distX = np.array(D_Kernels[0]) * np.array(alphaK[0,0])
        
        for i in range(1,len(D_Kernels)) :
            distX = distX + np.array(D_Kernels[i]) * alphaK[0,i]
        
        distX1 = np.sort(distX)
        inx = np.argsort(distX)

    LF = F_eig1
    D = np.repeat(0.0, S.shape[1]*S.shape[1]).reshape(S.shape[1],S.shape[1])
    np.fill_diagonal(D,np.sum(D,axis=0))
    L = D - S
    
    eigen_L = LA.eig(L)
    U = eigen_L[1]
    D = eigen_L[0]

    #print(S)
    if len(no_dim)== 1 :
        U_index = np.arange(U.shape[1]-no_dim+1, U.shape[1]+1)
        U_index = (-np.sort(-U_index))-1
        #print("S is ")
        #print(type(S))
        F_last = ts.tsne(S,no_dim[0], U[:,U_index])
        
    else:
        F_last = list()
        for i in range(len(no.dim)):
            U_index = np.arange(U.shape[1]-no_dim+1, U.shape[1]+1)
            U_index = (-np.sort(-U_index))-1
            F_last.append(ts.tsne(S,no_dim[i], U[:,U_index]))
    #print(U[:,U_index])
    #raise ValueError("stop")           
    y = kmeans(F_last,c)
   
    ydata = ts.tsne(np.matrix(S),2,None)
    print(S)
    results = list()
    #print("y is ")
    #print(y)
    results.append(y)
    #print("S is ")
    #print(S)
    results.append(S)
    #print("F_last  is ")
    #print(F_last)
    results.append("F_last is ")
    results.append(F_last)
    #print("ydata  is ")
    #print(ydata)
    results.append(ydata)
    #print("alphaK  is ")
    #print(alphaK)
    results.append(alphaK)
    #print("converge  is ")
    #print(converge)
    results.append(converge)
    #print("LF  is ")
    #print(LF)
    results.append(LF)
    return results
                             
        
def asvectorCol(w):
    v = w[:,0]
    for i in range(w.shape[1]-1):
        v = np.concatenate((v,w[:,i+1]),axis=0)
    return np.array(v.T)

def asvectorRow(w):
    v = np.array(w[0,:])[0]
    for i in range(w.shape[0]-1):
        v = np.concatenate((v,np.array(w[i+1,:])[0]),axis=0)
    return np.array(v)

if __name__=='__main__':
    data = sio.loadmat('/Users/Zeyin/Desktop/Study/Bioinformatics/SIMLR-SIMLR/data/Test_1_mECS.mat')
    data1 = data["in_X"]
    data1 = data1[0:181,0:8988].T
    data2 = np.repeat(3,1)
    res = SIMLR(data1,data2,data2)
    dd=res[4]
    print(dd)
    import matplotlib.pyplot as plt
    plt.plot(dd[:,0],dd[:,1],'*')
    plt.show()


