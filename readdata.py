import numpy as np

data = open("/Users/Zeyin/Desktop/Study/Bioinformatics/project/data1.txt",'r')
f=data.next()
k=0
res = list()
for line in data:
    if(k==4):
        break
    s = line.split(" ")
    l = len(s)
    curl = list()
    for i in range(1,l):
        curl.append(float(s[i]))
    res.append(res)
    k = k+1

data.close()

res = np.matrix(res)
print(res)
