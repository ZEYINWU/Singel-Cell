from numpy import linalg as LA

def eig1(A):
    eigen_A = LA.eig(A)
    v = eigen_A[1]
    d = eigen_A[0]

    d1 = -np.sort(-d)
    idx = np.argsort(-d)
    idx1 = idx 

    eigval = d[idx1]
    eigvec = v[,idx1].real
    eigval_full = d[idx]
    res=list()
    res.append(eigval)
    res.append(eigvec)
    res.append(eigval_full)sa
    return res
    
