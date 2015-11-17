from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=2, size_hint=10, dtype=COMPLEX64_T, is_symmetric=True)

b1 = np.array([1+1j], dtype=np.complex64)
b2 = np.array([1+1j, 12-1j], dtype=np.complex64)
b3 = np.array([1+1j, 1+1j, -1j], dtype=np.complex64)

l1.put_diagonal(-1, b3)

l1.print_to(sys.stdout)

########################################################################################################################
print "*" * 80

A = NewBandLLSparseMatrix(nrow=2, ncol=3, diag_coeff=[-1, 1, 2], numpy_arrays=[b1, b2, b3], dtype=COMPLEX64_T)

A.print_to(sys.stdout)

print "/" * 80
B = NewBandLLSparseMatrix(nrow=2, ncol=3, diag_coeff=[1], numpy_arrays=[b2], dtype=COMPLEX64_T)
print B