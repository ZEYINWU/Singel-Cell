import numpy as np
from scipy.stats import norm
import math
from handythread import foreach
import scipy.io as sio
## input: matrix x, and integer thread 
def multiKernel(x,threads):
    kernelType = list();
    kernelType.append("poly");

    kernelParams = list();
    kernelParams.append(0);

    N = x.shape[0];
    KK= 0;
    sigma = np.arange(2,0.8,-0.25);

    #compute the combinded kernels
    d = dist2(x,x);
    Diff = np.multiply(d,d);
    Diff_sort = (np.sort(Diff,axis=0)).T;

    # compute the combined kernels

    m = Diff.shape[0];
    n = Diff.shape[1];
    allk = np.arange(10,32,2);
    global D_Kernels;
    D_Kernels = list();
    def dkernels(l,x_fun=x,Diff_sort_fun=Diff_sort,allk_fun=allk,Diff_fun=Diff,sigma_fun=sigma,KK_fun=KK):
        if allk_fun[l]<((x_fun.shape[0])-1):
            TT = np.mean(Diff_sort_fun[:,np.arange(1,allk_fun[l]+1)],axis=1);
            TT = np.add(TT,np.finfo(float).eps);
            length = TT.shape[0]

            Sig = TT[:,0]
            for i in range(length-1):
                Sig = np.concatenate((Sig,Sig[:,0]),axis = 1)

            Sig = np.add(Sig,Sig.T);
            Sig = np.multiply(Sig,0.5);
            #print(Sig)
            Sig_valid = Sig > np.finfo(float).eps;
            Sig = np.add(np.multiply(Sig,Sig_valid),np.finfo(float).eps);
            for j in range(len(sigma_fun)):
                W = norm.pdf(Diff_fun,0,np.multiply(sigma_fun[j],Sig));
                D_Kernels.insert(KK_fun +l+j,np.matrix(np.multiply(np.add(W,W.T),0.5)))
    r = range(len(allk));
    foreach(dkernels,r,threads=threads);

    #print(D-Kernels)

    for i in range(len(D_Kernels)):
        K= D_Kernels[i];
        k = 1.0 / np.sqrt(np.diagonal(K)+1)
        G = np.array(K) * np.array(np.dot(np.matrix(k).T,np.matrix(k)))
        diag = np.diag(G)
        G1 = np.repeat(0.0,(len(diag)*len(diag))).reshape(len(diag),len(diag))
        for j in range(len(diag)):
            G1[:,j] = diag

        G2 = G1.T

        D_Kernels_tmp = (np.array(G1)+np.array(G2) - 2.0*np.array(G)) / 2.0
        lll = len(np.diag(D_Kernels_tmp))
        newd = np.repeat(0.0,lll*lll).reshape(lll,lll)
        np.fill_diagonal(newd,np.diag(D_Kernels_tmp)) #### return void
        D_Kernels_tmp = np.array(D_Kernels_tmp) - np.array(newd)

        D_Kernels[i] = D_Kernels_tmp;
     
    return D_Kernels;            
         
        
    


def dist2(x,c):
    n1 = x.shape[0];
    d1 = x.shape[1];
    n2 = c.shape[0];
    d2 = c.shape[1];

    A = np.matrix(np.repeat(1,n2)).T;
    B = np.sum(np.multiply(x,x).T,axis=0);
    C = np.matrix(np.repeat(1,n1)).T;
    D = np.sum(np.multiply(c,c).T,axis=0) ;
    E = np.multiply(np.dot(x,c.T),-2);

    F = np.dot(np.matrix(A),np.matrix(B)).T
    G = np.dot(np.matrix(C),np.matrix(D))
    dist = np.add(np.add(F,G),E)

    return dist ;

if __name__=='__main__':
    data = sio.loadmat('/Users/Zeyin/Desktop/Study/Bioinformatics/SIMLR-SIMLR/data/Test_1_mECS.mat')
    data1 = data["in_X"]
    data1 = data1[0:13,0:13].T
    test = np.matrix(data1)
    print ((multiKernel(test,1)))
    



    
