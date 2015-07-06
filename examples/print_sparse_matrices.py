from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLinearFillLLSparseMatrix(nrow=12, ncol=6, size_hint=40, store_zeros=False)
l1.clear_submatrix(1,3, 4,6)
print l1
print l1.to_csr()
print l1.to_csc()

print "*" * 80

l1.print_to(sys.stdout)

print "v" * 80

print l1.T
print l1.to_csr().T

print "w" * 80

l2 = NewLinearFillLLSparseMatrix(first_element=0+5j, nrow=12, ncol=6, size_hint=40, store_zeros=True, dtype=types.COMPLEX64_T)

print l2.H
print l2.to_csr().H

print "z" * 80

print l2.conj
print l2.to_csc().conj


print "/" * 80
print "/" * 80

l1 = NewLinearFillLLSparseMatrix(nrow=102, ncol=600, size_hint=40000, store_zeros=False)
l1.clear_submatrix(1,3, 4,6)
print l1
