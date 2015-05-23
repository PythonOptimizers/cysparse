import time
from pysparse.sparse import spmatrix

n = 500000
size_hint = 200000

A = spmatrix.ll_mat(n, n, size_hint)

for i in xrange(size_hint):
    A[i, i] = i
    A[i+2, i* 2] = 10 * i

# refill

for i in xrange(size_hint/100):
    for j in xrange(size_hint/100):
        A[i, j] = i