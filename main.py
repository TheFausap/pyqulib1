import cmath
import math
import scipy.sparse as sp
import numpy as np

NUM = bin(23)[2:]
NUM = NUM[::-1] # reverse the input binary digits
Ncoeff = 1.0/math.sqrt(2**len(NUM))
pCoeff = 1.0/math.sqrt(2)
a = []
t = 0.0
_k0 = np.array([1,0],dtype=complex)
_k0.shape=(2,1)
_k1 = np.array([0,1],dtype=complex)
_k1.shape=(2,1)
k0 = sp.coo_matrix(_k0)
k1 = sp.coo_matrix(_k1)

for i in range(1, len(NUM) + 1):
    for j in range(1, i + 1):
        # print(i,"-",j,"==",NUM[j-1])
        expon = 2**(i-j+1) #should be negative
        t += int(NUM[j - 1]) * (2 ** -(i-j+1))
        # print("expon:",expon,"t: ",t)
    t = cmath.exp(2 * cmath.pi * complex(0, 1) * t)
    a.append(t)
    t = 0.0

z1 = pCoeff*sp.coo_matrix(_k0+a[0]*_k1)

for i in a[1:]:
    #print("Element with other a: ",i)
    z1 = sp.kron(z1,pCoeff*(k0 + i*k1))

print((z1).shape[0])

