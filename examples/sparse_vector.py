from cysparse.sparse.ll_mat import *

v = LLSparseMatrix(nrow=4, ncol=1)
v.put_triplet([0, 2], [0, 0], [1.0, 2.0])

A = LinearFillLLSparseMatrix(nrow=3, ncol=4)

print v
print A

C = A * v
print C
print matvec(A, v)


B = LinearFillLLSparseMatrix(nrow=2, ncol=2)

import numpy as np

a=np.mat('4 3; 2 1', dtype=np.float64)

print a

print type(a)
D = B * a

print D
print type(D)

print "+" * 80

A = LinearFillLLSparseMatrix(nrow=2, ncol=2)
B = np.mat('4 3; 2 1', dtype=np.float64) # NumPy matrix

print A
print B

C = A * B

print C
print type(C)

print "=" * 80

