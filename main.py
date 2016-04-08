import cmath
import math
import scipy.sparse as sp
import numpy as np
import time

NUM = bin(2333888)[2:]
NUM = NUM[::-1]  # reverse the input binary digits
pCoeff = 1.0/math.sqrt(2)
a = []
t = 0.0
_k0 = np.array([1, 0], dtype=complex)
_k0.shape = (2, 1)
_k1 = np.array([0, 1], dtype=complex)
_k1.shape = (2, 1)
k0 = sp.coo_matrix(_k0)
k1 = sp.coo_matrix(_k1)

t0 = time.time()

for i in range(1, len(NUM) + 1):
    for j in range(1, i + 1):

        expon = 2**(i-j+1)  # should be negative
        t += int(NUM[j - 1]) * (2 ** -(i-j+1))

    t = cmath.exp(2 * cmath.pi * complex(0, 1) * t)
    a.append(t)
    t = 0.0

z1 = pCoeff*sp.coo_matrix(_k0+a[0]*_k1)

for i in a[1:]:
    z1 = sp.kron(z1, pCoeff*(k0 + i*k1))

print("Elapsed: ", time.time()-t0)
print("Number of input qubit: ", len(NUM))
print("QUREG length: ", z1.shape[0])

