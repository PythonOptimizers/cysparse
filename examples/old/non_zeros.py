from cysparse.sparse.ll_mat import *
from cysparse.sparse.s_mat import *

import cysparse.common_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=10, ncol=10, size_hint=40, store_zeros=True)
print l1
print type(l1)             # lots of classes are used internally...

l1[2, 2] = 450000000000000000000  # huge number
l1[9, 9] = np.inf
l1[0, 0] = np.nan

l1.put_triplet([1,1], [1, 2], [5.6, 6.7])  # i, j, val

l1[4, 5] = 98374983.093843483

l1[0, 1] = 0.0

l1.print_to(sys.stdout)



########################################################################################################################
print '*' * 80

print l1.nnz

l1[0, 2] = 0.0

print l1.nnz

print l1.find()

with NonZeros(l1):
    l1[0, 3] = 0.0
    l1[0, 4] = 0.00001

print l1.nnz

l1[0,5] = 0.0


print l1.nnz