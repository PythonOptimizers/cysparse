import time
from cysparse.sparse.ll_mat import NewLLSparseMatrix

n = 500000
size_hint = 200000

A = NewLLSparseMatrix(size=n, size_hint=size_hint)

for i in xrange(size_hint):
    A[i, i] = i
    A[i+2, i* 2] = 10 * i

# refill

for i in xrange(size_hint/100):
    for j in xrange(size_hint/100):
        A[i, j] = i