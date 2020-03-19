import random
import numpy as np
import matplotlib.pyplot as plt
import pylab


def quantum_walk(a: np.array):
    N = 100  # number of random steps
    P = 0   # number of positions
    for row in a:
        P += 1 if not np.any(row) else 0
    # quantum coin
    coin0 = np.array([1, 0])  # |0>
    coin1 = np.array([0, 1])  # |1>

    # hadamard coin operator
    C00 = np.outer(coin0, coin0)  # |0><0|
    C01 = np.outer(coin0, coin1)  # |0><1|
    C10 = np.outer(coin1, coin0)  # |1><0|
    C11 = np.outer(coin1, coin1)  # |1><1|

    C_hat = (C00 + C01 + C10 - C11) / np.sqrt(2.)

    # shift (step) operator
    ShiftPlus = np.roll(np.eye(P), 1, axis=0)
    ShiftMinus = np.roll(np.eye(P), -1, axis=0)
    S_hat = np.kron(ShiftPlus, C00) + np.kron(ShiftMinus, C11)

    # walk operator
    U = S_hat.dot(np.kron(np.eye(P), C_hat))

    # initial state
    posn0 = np.zeros(P)
    posn0[N] = 1  # array indexing starts from 0, so index N is the central posn
    psi0 = np.kron(posn0, (coin0 + coin1 * 1j) / np.sqrt(2.))

    # state after N steps
    psiN = np.linalg.matrix_power(U, N).dot(psi0)

    # measurement operator
    prob = np.empty(P)
    for k in range(P):
        posn = np.zeros(P)
        posn[k] = 1
        M_hat_k = np.kron(np.outer(posn,posn), np.eye(2))
        proj = M_hat_k.dot(psiN)
        prob[k] = proj.dot(proj.conjugate()).real

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(P), prob)
    plt.plot(np.arange(P), prob, 'o')
    loc = range(0, P, P // 10)  # Location of ticks
    plt.xticks(loc)
    plt.xlim(0, P)
    ax.set_xticklabels(range(-N, N + 1, P // 10))

    plt.show()