import numpy as np 
import re

data = np.loadtxt('data/graphs_10.csv')
length = len(data)
#data = data[:-2]
#print(data)
#data = np.reshape(data,[-1,10,10])
#print(data)

#Getting the matrix dimensions from the file name

for i in range(length-999):
    A = data[i]
    A = A[:-4]
    A = np.reshape(A,[10,10])
    print(A,A[0])
    for j in range(10):
        for k in range(10):
            L = A[j][k] + A[k][j]
            #print(L) 
        L = L - 2*A[j][j]
        print(L)

print(L)    



regex = re.compile(r'\d+')
n = [int(x) for x in regex.findall('data/graphs_10.csv')][0]

print(n)
def edge_to_vertex(data_file):

    n = [int(x) for x in regex.findall(data_file)][0]
    data = np.loadtxt(data_file)
    length = len(data)

    data = np.reshape(data,[-1,1,n,n])


    #for i in range 

#print(length,data[0][0])