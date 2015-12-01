# test CSR multiplied by CSC
from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

A = NewLinearFillLLSparseMatrix(size=2)

print A

A_csr = A.to_csr()

print A_csr
print A_csr.to_ndarray()

B = NewLinearFillLLSparseMatrix(size=2, row_wise=False)

print B

B_csc = B.to_csc()

print B_csc
print B_csc.to_ndarray()

print "=" * 80

print A * B

print A_csr * B_csc

print np.dot(A_csr.to_ndarray(), B_csc.to_ndarray())