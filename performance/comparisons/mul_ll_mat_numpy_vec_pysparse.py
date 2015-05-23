from pysparse.sparse import spmatrix
import numpy as np

n = 500000000
size_hint = 200000

A = spmatrix.ll_mat(n, n, size_hint)

for i in xrange(size_hint):
    A[i, i] = i
    A[i+2, i* 2] = 10 * i

# refill

for i in xrange(size_hint/100):
    for j in xrange(size_hint/100):
        A[i, j] = i

b = np.arange(0, n, dtype=np.float64)
c = np.empty(n, dtype=np.float64)
A.matvec(b, c)

print c

print c[1]
print c[4578]
print c[3463463]