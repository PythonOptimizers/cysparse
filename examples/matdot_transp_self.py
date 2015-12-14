from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=10, dtype=types.COMPLEX64_T)

for i in xrange(3):
    for j in xrange(4):
        if i > j:
            l1[i, j] = (i + j) / 3 + 1

l1.print_to(sys.stdout)

A = l1.matdot_transp_self()

print "9" * 80
print A
A.print_to(sys.stdout)

print "*" * 80

d = np.array([1, 2, 3, 4], dtype=np.complex64)

B = l1.matdot_transp_self(d)

B.print_to(sys.stdout)
