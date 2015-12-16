from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX64_T)

l1.put_triplet([1, 1, 1], [0, 1, 2], [1+1j, 1-1.0j, 2+1j])  # i, j, val

print "*" * 30 + " Original matrix: " + "*" * 30
l1.print_to(sys.stdout)

print
print "REAL MATRICES"
print "*" * 30 + " Transpose matrix: " + "*" * 30
l1_T = l1.create_transpose()
l1_T.print_to(sys.stdout)

print "*" * 30 + " Conjugate transpose matrix: " + "*" * 30
l1_conj = l1.create_conjugate()
l1_conj.print_to(sys.stdout)

print "*" * 30 + " Conjugate transpose matrix: " + "*" * 30
l1_H = l1.create_conjugate_transpose()
l1_H.print_to(sys.stdout)

b2 = np.array([1+1j, 1+1j], dtype=np.complex64)
b3 = np.array([1+1j, 1+1j, -1j], dtype=np.complex64)

########################################################################################################################
print "=" * 80

print "Surprise proxy:"

surprise_proxy = l1.H.H.T.conj.H.T
print surprise_proxy

print surprise_proxy * b2

print "Real matrix"
print l1_T * b2



