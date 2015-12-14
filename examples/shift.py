from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=4, ncol=3, size_hint=10, dtype=COMPLEX256_T, itype=INT64_T)
l2 = NewLLSparseMatrix(nrow=4, ncol=3, size_hint=10, dtype=COMPLEX256_T, itype=INT64_T)

for i in xrange(4):
    l1[i, 1] = 1

l1.print_to(sys.stdout)

for i in xrange(3):
    l2[1, i] = 1

l2.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

sigma = 10 + 10.01j
l1.shift(sigma, l2)

l1.print_to(sys.stdout)

