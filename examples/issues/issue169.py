from cysparse.sparse.ll_mat import *
import numpy as np

A = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=6)
A.put_triplet(np.array([0, 0, 1, 1, 1]), np.array([0, 1, 0, 1, 2]), np.array([2, 1, 3, 4, 5], 'd'))

print "A:"
print A

B = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=8)
print "B:"
print B

B[:2,:3] = A[:, :]

print "B[:2,:3] = A[:, :]; print B"
print B

print "-" * 80

B[2:3, :] = A[0,0]
print "B[2:3, :] = A[0,0]; print B"
print B

print "=" * 80


B[2:3,:3] = A[1:2, :]
print "B[2:3,:3] = A[1:2, :], print B"
print B

print "&" * 80
# this should raise an exception...
#B[2:3,:] = A[1:2, :]

G = np.ones((2,1))
print G

B[1:3, 2] = G

print B