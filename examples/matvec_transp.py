from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=10)

l1[2, 2] = 45
l1[0, 2] = 2
l1[0, 0] = 23.5

l1.put_triplet([1,1], [1, 2], [5.6, 6.7])  # i, j, val
l1.print_to(sys.stdout)

a = np.array([1,1,1], dtype=np.float64)

print "and now..."

c = l1.T * a

print "tadaaa..."
print c
