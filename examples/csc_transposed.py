from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=10, ncol=5, size_hint=40)



l1[2, 2] = 45
l1[3, 2] = np.inf
l1[0, 0] = np.nan

l1.put_triplet([1,1], [1, 2], [5.6, 6.7])  # i, j, val
print l1

l1[3, 1] = 98374983.093843483

l1.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

l1_csc = l1.to_csc()

l1_csc.debug_print()

l1_csc.print_to(sys.stdout)

A = l1_csc.T

print A
print A.T

n = np.ones(10, dtype=np.float64)

print A * n