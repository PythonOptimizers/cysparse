from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX64_T)

l1.put_triplet([1, 1,1], [0, 1, 2], [8+8j, 5.6-1.0j, 6.7+3.2j])  # i, j, val

l1.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

H = l1.H

print H

HH = H.H

print HH

for i in xrange(3):
    for j in xrange(2):
        print H[i, j]

b = np.array([1+1j, 1+1j, 1+1j, 1+1j], dtype=np.complex128)

print H.matvec(b)