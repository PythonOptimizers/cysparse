from cysparse.sparse.ll_mat import NewLLSparseMatrix
from cysparse.types.cysparse_types import INT32_T, INT64_T, FLOAT64_T
import numpy as np

n = 10
size_hint = 6

use_float = False

if use_float:
    A = NewLLSparseMatrix(size=n, size_hint=size_hint, dtype=FLOAT64_T)
else:
    A = NewLLSparseMatrix(size=n, size_hint=size_hint, dtype=INT64_T)


for i in xrange(size_hint):
    A[i, i] = i
    A[i+2, i] = 10 * i


# refill

for i in xrange(size_hint):
    for j in xrange(size_hint):
        A[i, j] = i

for i in xrange(n):
    for j in xrange(n):
        print "%d " % A[i,j],
    print

if use_float:
    b = np.arange(0, n, dtype=np.float64)
else:
    b = np.arange(0, n, dtype=np.int64)


#c = np.empty(n, dtype=np.float64)

c = A * b

print c

