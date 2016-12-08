import numpy as np
import network as nw
import multikernel as mk
from numpy import linalg as LA
import scipy.io as sio
import util_silmr as us
import tsne as ts
from scipy.cluster.vq import vq, kmeans, whiten
import project as pj
import SILMR as silmr

data = sio.loadmat('/Users/Zeyin/Desktop/Study/Bioinformatics/SIMLR-SIMLR/data/Test_1_mECS.mat')
data1 = data["in_X"]
data1 = data1[0:181,0:8988].T
data2 = np.repeat(3,1)
res = silmr.SIMLR(data1,data2,data2)
dd = res[4]
#import matplotlib.pyplot as plt
#plt.plot(np.array(dd[:,0]),'*',dd[:,1],'.',dd[:,2],'+')
#plt.show()
