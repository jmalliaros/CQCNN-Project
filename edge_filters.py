import numpy as np 
import re

#just testing
data = 'data/graphs_10.csv'

#Getting the matrix dimensions from the file name
regex = re.compile(r'\d+')
def matrix_size(data_file):
    n = [int(x) for x in regex.findall(data_file)][0]
    return n 

#Length of data (i.e., 1000)
def length_data(data_file):
    data = np.loadtxt(data_file)
    return len(data)

#Getting adjacency matrix of one graph
def data_to_graph(data_file, i):
    data = np.loadtxt(data_file)
    A = data[i]
    A = A[:-4]
    A = np.reshape(A,[10,10])
    return A

#Summation in edge-to-edge filtering
def sum(M,n,i,j):
    x = 0
    for k in range(n):
        x += M[i][k] + M[k][j]
    return x

#Summation for edge-to-vertex filtering
def sum_ev(M,n,i):
    x = 0
    for k in range(n):
        x += M[i][k] + M[k][i]
    return x

#Edge to edge filter, initiate empty array first
def edge_to_vertex(F,n):
    Fev = np.zeros([1,n])[0]
    for i in range(n):
        Fev[i] = sum_ev(F,n,i) - 2*F[i][i]
    return Fev

   #for i in range 

#Looping through all graphs in data file 
def edge_to_edge(data_file):
    length = length_data(data_file)
    n = matrix_size(data_file)
    Fee = np.zeros([n,n])

    for x in range(length-950):
        M = data_to_graph(data_file,x)
        for i in range(n):
            for j in range(n):
                Fee[i][j] = (sum(M,10,i,j) - 2*M[i][j])*M[i][j]
                Fev = edge_to_vertex(Fee,n)
        print(Fev)
    return Fev
        #print("graph:\n", M,"\n", "EtoE filter: \n", Fee, "EtoV filter: \n", Fev)    

print(edge_to_edge(data))

