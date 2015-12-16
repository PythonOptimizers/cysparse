from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=10)

l1[2, 2] = 45
l1[0, 2] = 2
l1[0, 0] = 23.5

l1.put_triplet([1,1], [1, 2], [5.6, 6.7])  # i, j, val
l1.print_to(sys.stdout)

a = np.array([1,1,1], dtype=np.float64)

print "=" * 80
T = l1.T
print T.nrow

F = l1.T.T.T.T.T
print F.dtype

c = F * a
print c


print "=" * 80

b = np.array([1, 1, 1, 1], dtype=np.float64)

d = l1.T.T.T.T * b

print d

print "=" * 40


print l1.T[0, 2]
print l1.T[2, 0]

