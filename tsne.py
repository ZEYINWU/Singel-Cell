import numpy as np
import scipy.io as sio
import random
def tsne(X,k,initial_config):

    max_iter = 3000
    min_cost = 0.0
    epoch = 100

    momentum = 0.8
    final_momentum = 0.7
    mom_switch_iter = 250

    epsilon = 500.0
    min_gain = 0.01
    initial_P_gain = 4.0

    n = (X.shape[0])
    
    eps = np.finfo(float).eps

    #if np.isnan(initial_config):
#    ydata = initial_config
#    initial_P_gain = 1
    #else:
 #   ydata = np.matrix(np.random.normal(0.0,1.0,k*n)).reshape(n,k)
    #print("ydata is ")
    #print(ydata)
 #   ydata = np.matrix("0.6571193, -1.9158855, -0.2965425;0.3868631,  0.7200120,  0.4099676;0.9175513, -0.8469215,  0.5158201")

    #if (initial_config) is not None:
#        ydata = initial_config
        
#        initial_P_gain = 1
#    else:
    
    if initial_config is not None and initial_config.shape[0]!=1:
        ydata = initial_config
        initial_P_gain = 1
#        ydata = np.matrix(np.random.normal(0.0,1.0,k*n)).reshape(n,k)
 #       ydata=np.matrix("-1.03811316 , 0.2717166 , 0.6244905,  1.6301632 , 0.1038819;1.39669953, -0.5014884 , 0.7745979 , 1.2951793, -0.6959245; 0.61294758 ,-0.3976206 , 0.0395659,  1.7197544 ,-0.5571205;-1.07945380 , 0.3464190,  1.5074215, -0.1667624, -0.6581690; 0.47860815 , 0.9104152, -0.1426967, -0.1028931 ,-1.1953642; 0.13212600 , 1.0683870,-1.3965235,  0.6636401, -1.6036982; -0.07440915 ,-0.7048408 ,-1.3499397, -1.3138075 , 1.0891606; -1.40368928 , 0.5488038 , 0.8437093 ,-0.9947427, -0.7602520;-1.30680392 , 0.4817890 ,-0.4378943 ,-1.2450627 ,-0.3726204;-0.21070500 , 0.4678479,  0.4290807,  0.6702021,  0.1439408;0.66847317, -0.92297959, -1.15694119, -0.71589809,  0.3469709;0.64940661, -1.74822941, -0.13348036,  0.55788373,  0.45194675;1.16919698,  0.51906695,  0.08935842,  0.4436435 , -1.34942478")
         
    else:
        ydata = np.matrix(np.random.normal(0.0,1.0,k*n)).reshape(n,k)
        
    #print(ydata)
    P = X
    P = 0.5 * (P+P.T)

    P[P<eps] = eps
    P = P / np.sum(P)

    P = P * initial_P_gain
    #print(ydata)
    grads = np.repeat(0.0,ydata.shape[0]*ydata.shape[1]).reshape(ydata.shape[0],ydata.shape[1])
    incs = np.repeat(0.0,ydata.shape[0]*ydata.shape[1]).reshape(ydata.shape[0],ydata.shape[1])
    gains = np.repeat(1.0,ydata.shape[0]*ydata.shape[1]).reshape(ydata.shape[0],ydata.shape[1])
    
    #print("P is ")
    #print(P)
    #print(ydata)
    Q = P
    for iter in range(max_iter):
        if iter % epoch ==0.0:
            cost = np.sum(np.sum(np.array(P)*np.array(np.log((P+eps)/(Q+eps))),axis=1))
            print("Iteration #"+str(iter) + ": cost is " +str(cost))
            if cost < min_cost :
                break

        sum_ydata = np.sum(np.multiply(np.array(ydata),np.array(ydata)),axis=1)
        
        num = 1.0 /(1.0 + np.matrix(sum_ydata).T + (np.dot(np.matrix(-2 * ydata),np.matrix(ydata.T))+np.matrix(sum_ydata)))
        
        np.fill_diagonal(num,0.0)

        Q =num / np.sum(np.sum(num))    ################adjust with 0.45
        
        Q[Q<eps] = eps

        
        
        stiffnesses = np.array(P-Q) * np.array(num)
        
        mo = np.repeat(0.0,stiffnesses.shape[0]*stiffnesses.shape[1]).reshape(stiffnesses.shape[0],stiffnesses.shape[1])
        np.fill_diagonal(mo,np.sum(stiffnesses,axis=0))
        
        
        grads = 4*np.dot(np.matrix((mo - stiffnesses)),np.matrix(ydata))
        
        
        check1 = (np.sign(grads) != np.sign(incs))
        check1[check1==True] = 1.0
        check1[check1==False] =0.0
        
        check2 = (np.sign(grads) == np.sign(incs))
        check2[check2==True] = 1.0
        check2[check2==False] =0.0
        gains = np.array(gains + 0.2) * np.array(check1) + np.array(gains) * 0.8 *np.array(check2)
        gains[gains < min_gain] = min_gain
        
        incs = (momentum) * (incs) - epsilon * (np.array(gains)*np.array(grads)) 
        
        
        ydata = ydata + incs
        ydata = ydata - np.mean(ydata,axis=0)
        
        ydata[ydata < -100] = -100.0
        ydata[ydata > 100] = 100.0
        if iter == mom_switch_iter :
            momentum = final_momentum

        if iter == 100.0 and initial_config is None:
            P = P/4.0
       


    return ydata


if __name__ == "__main__":
    data = sio.loadmat('/Users/Zeyin/Desktop/Study/Bioinformatics/SIMLR-SIMLR/data/Test_1_mECS.mat')
    data1 = data["in_X"]
    data1 = data1[0:10,0:10].T
    X = np.matrix("1,2,3;4,5,6;7,8,9")
    k=3
    initial_config = np.matrix(np.repeat(5,5))
    res = tsne(np.matrix(data1),5,initial_config)
    print res


























        









    
