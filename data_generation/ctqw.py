import numpy as np
from odeintw import odeintw
import matplotlib.pyplot as plt

#example adj matrix
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

one_ket = np.array([0, 1])  # |1>
rho0 = np.outer(one_ket, one_ket)  # |1><1|

t = np.linspace(0.0, 10.0, 100)

def GKSL(rho, t, A):
    h_bar = 6.582119569e-16
    gamma = 1
    n = np.size(A,0)

    A = np.column_stack((A, np.zeros(n)))
    A = np.vstack ((A, np.zeros(n+1)))
    H = h_bar*A

    n_bra = np.zeros(n)
    n_bra[n-1] = 1
    n_1_ket = np.zeros(n + 1).T
    n_1_ket[n] = 1
    print(n_bra, n_1_ket)
    L = np.matrix(np.outer(n_1_ket, n_bra))
    print(L.getH())
    L_dag_L = np.matmul(L.getH(), L)

    drho_dt = -(1j/h_bar)*(np.matmul(H, rho) - np.matmul(rho, H)) + gamma*(np.matmul(L, np.matmul(rho, L.getH())) - 0.5*(np.matmul(L_dag_L, rho) + np.matmul(rho, L_dag_L)))

    return drho_dt

p_th = 1/np.log(np.size(A,0))
rho = odeintw(GKSL, rho0, t, args=(A,))

plt.plot(t, rho, 'b', t, p_th, 'g--')
plt.xlabel('time')
plt.ylabel('rho(t)')
plt.savefig("mygraph.png")
