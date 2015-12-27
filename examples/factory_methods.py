from cysparse.sparse.ll_mat import *
#from cysparse.common_types.cysparse_types import *

A = LLSparseMatrix(nrow=2, ncol=3, size_hint=3)

A = LLSparseMatrix(nrow=256, ncol=3398, size_hint=600)

A = LLSparseMatrix(size=5578, size_hint=600, itype=INT32_T, dtype=COMPLEX128_T)

print A

A[1,2] = 56.97

print A

print A[1,2]

B = LLSparseMatrix(size=2, dtype=FLOAT64_T)

B.put_triplet([0, 0, 1], [0, 1, 0], [1, 2, 40.6])

print B

print B[1, 0]

print B.nrow
print B.ncol

print "=" * 80

C = LLSparseMatrix(mm_filename='bcsstk01.mtx', itype=INT64_T, dtype=FLOAT64_T)

print C

D = LLSparseMatrixFromMMFile('bcsstk01.mtx')

print D
print type(D)

E = LLSparseMatrixFromMMFile('bcsstk01.mtx', store_zero=True, test_bounds=False)

print E
print type(E)

print "+" * 80

F = A = IdentityLLSparseMatrix(size = 3, dtype=COMPLEX64_T)
print F

G = DiagonalLLSparseMatrix(element=3-5j, nrow=2, ncol=3, dtype=COMPLEX128_T)
print G

print ")" * 80

import numpy as np

l1 = LLSparseMatrix(nrow=2, ncol=2, size_hint=10, dtype=COMPLEX64_T, is_symmetric=True)

b1 = np.array([1+1j], dtype=np.complex64)
b2 = np.array([1+1j, 12-1j], dtype=np.complex64)
b3 = np.array([1+1j, 1+1j, -1j], dtype=np.complex64)

#l1.put_diagonal(-1, b3)

print l1

########################################################################################################################
print "*" * 80

l2 = BandLLSparseMatrix(nrow=2, ncol=3, diag_coeff=[-1, 1, 2], numpy_arrays=[b1, b2, b3], dtype=COMPLEX64_T)

print l2

print "/" * 80
l3 = BandLLSparseMatrix(nrow=2, ncol=3, diag_coeff=[1], numpy_arrays=[b2], dtype=COMPLEX64_T)
print l3
