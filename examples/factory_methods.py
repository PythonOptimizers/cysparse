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

