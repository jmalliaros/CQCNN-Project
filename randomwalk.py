import random
import numpy as np
import matplotlib.pyplot as plt
import pylab

# one dimension classical random walk graph plot
# implemented from https://www.geeksforgeeks.org/random-walk-implementation-python/
def one_dim_rand_walk():
    # Probability to move up or down
    prob = [0.05, 0.95]

    # statically defining the starting position
    start = 2
    positions = [start]

    # creating the random points
    rr = np.random.random(1000)
    downp = rr < prob[0]
    upp = rr > prob[1]

    for idownp, iupp in zip(downp, upp):
        down = idownp and positions[-1] > 1
        up = iupp and positions[-1] < 4
        positions.append(positions[-1] - down + up)

    # plotting down the graph of the random walk in 1D
    plt.plot(positions)
    plt.show()

# Two dimension classical random walk graph plot
# implemented from https://www.geeksforgeeks.org/random-walk-implementation-python/
def two_dim_rand_walk():
    # defining the number of steps
    n = 100000

    # creating two array for containing x and y coordinate
    # of size equals to the number of size and filled up with 0's
    x = np.zeros(n)
    y = np.zeros(n)

    # filling the coordinates with random variables
    for i in range(1, n):
        val = random.randint(1, 4)
        if val == 1:
            x[i] = x[i - 1] + 1
            y[i] = y[i - 1]
        elif val == 2:
            x[i] = x[i - 1] - 1
            y[i] = y[i - 1]
        elif val == 3:
            x[i] = x[i - 1]
            y[i] = y[i - 1] + 1
        else:
            x[i] = x[i - 1]
            y[i] = y[i - 1] - 1

    # plotting stuff:
    pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
    pylab.plot(x, y)
    pylab.savefig("rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600)
    pylab.show()


# quantum walk graph plot
# implemented from https://susan-stepney.blogspot.com/2014/02/mathjax.html?fbclid=IwAR2LzjdRMWKs9yV8pUHU7_5BaPGBnuoXyKZN1TbyrWSb-zARBSIMf6OyJG8
def quantum_walk():
    N = 100  # number of random steps
    P = 2 * N + 1  # number of positions

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
if __name__ == '__main__':
    quantum_walk()