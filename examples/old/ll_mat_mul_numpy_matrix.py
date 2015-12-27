from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

import sys

#############
print '*' * 80
print "A * B"
l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=6)
l1.put_triplet([0, 0, 1, 1, 1], [0, 1, 0, 1, 2], [2, 1, 3, 4, 5])

l1.print_to(sys.stdout)
print l1.nnz

n = np.array([[1, 2], [3, 4] , [5, 6]], dtype=np.float64)
print n
C = l1 * n

print C

#############
print '*' * 80
print "A * B with A symmetric"

l2 = NewLLSparseMatrix(nrow=3, ncol=3, size_hint=6, is_symmetric=True)
l2.put_triplet([0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2 ], [0, 3, 1, 4, 5, 2 ])

l2.print_to(sys.stdout)

D = l2 * n

print D


#############
print '*' * 80
print "A^t * B"

m = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float64)


E = l1.matdot_transp(m)

print E

#############
print '*' * 80
print "A^t * B with A symmetric"



F = l2.matdot_transp(n)

print F

