from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=10)

for i in xrange(3):
    for j in xrange(4):
        if i > j:
            l1[i, j] = (i + j) / 3 + 1

l1.print_to(sys.stdout)

l2 = NewLLSparseMatrix(nrow=3, ncol=2, size_hint=10)

for i in xrange(3):
    for j in xrange(2):
        if i <= j:
            l2[i, j] = (i + j) / 2 + 1

l2.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

C = l1.matdot_transp(l2)

C.print_to(sys.stdout)