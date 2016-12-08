import numpy as np
import scipy.io as sio
def projsplx(y,x):

    m = y.shape[0]
    n = y.shape[1]

    s = np.repeat(0.0,m).reshape(m,1)
    vs = np.repeat(0.0,m).reshape(m,1)

    for k in range(n):

        means = 0.0
        mins = 100000.0

        for j in range(m):
            s[j,0] = y[j,k]
            means = means + s[j,0]
            if mins > s[j,0]:
                mins = s[j,0]

        for j in range(m):
            s[j,0] = s[j,0] - (means - 1.0)/m

        ft = 1;

        if mins<0:
            f = 1
            lambda_m = 0
            while (np.absolute(f) > 1e-10):
                npos = 0
                f = 0
                for j in range(m):
                    vs[j,0] = s[j,0] - lambda_m

                    if vs[j,0] >0 :
                        npos = npos + 1
                        f = f+ vs[j,0]
                lambda_m = lambda_m + (f-1)/npos

                if ft>100 :
                    for j in range(m):
                        if vs[j,0] > 0:
                            x[j,k] = vs[j,0]
                        else:
                            x[j,k] = 0
                    break
                ft = ft + 1

            for j in range(m):
                if vs[j,0]>0 :
                    x[j,k] = vs[j,0]
                else:
                    x[j,k] = 0

        else:
            for j in range(m):
                x[j,k] = s[j,0]


    return x.T



if __name__ == "__main__":
    data = sio.loadmat('/Users/Zeyin/Desktop/Study/Bioinformatics/SIMLR-SIMLR/data/Test_1_mECS.mat')
    data1 = data["in_X"]
    data1 = data1[0:20,0:20].T
    x=data1
    y=-data1
    res = projsplx(x,y)
    print(res)


                
        
