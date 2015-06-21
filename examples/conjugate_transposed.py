from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX64_T)

l1.put_triplet([1, 1, 1], [0, 1, 2], [1+1j, 1-1.0j, 2+1j])  # i, j, val

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

b = np.array([1+1j, 1+1j], dtype=np.complex64)

print H.matvec(b)

########################################################################################################################
print "=" * 80

l2 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX64_T, __is_symmetric=True)
l2.put_triplet([1, 1], [0, 1], [1+1j, 1-1.0j])  # i, j, val

l2.print_to(sys.stdout)

H2 = l2.H

print H2

print H2.matvec(b)


########################################################################################################################
print "=" * 80

l3 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX256_T)
l3.put_triplet([1, 1, 1], [0, 1, 2], [1+1j, 1-1.0j, 2+1j])  # i, j, val

l3.print_to(sys.stdout)

H3 = l3.H

print H3

c = np.array([1+1j,0, 1+1j, 0], dtype=np.complex256)

print H3.matvec(c[::2])

########################################################################################################################
print "=" * 80

l4 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX256_T, __is_symmetric=True)
l4.put_triplet([1, 1], [0, 1], [1+1j, 1-1.0j])  # i, j, val

l4.print_to(sys.stdout)

H4 = l4.H

print H4

print H4.matvec(c[::2])

