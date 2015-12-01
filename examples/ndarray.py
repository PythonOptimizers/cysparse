from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=10, dtype=types.COMPLEX256_T)

for i in xrange(3):
    for j in xrange(4):
        if i > j:
            l1[i, j] = (i + j) / 3 + 1

l1.print_to(sys.stdout)

print l1.to_ndarray()

########################################################################################################################
print "*" * 80

print l1.to_csc().to_ndarray()

########################################################################################################################
print "*" * 80

print l1.to_csr().to_ndarray()
