from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

A = LinearFillLLSparseMatrix(nrow=4, ncol=3, dtype=INT32_T, itype=INT64_T)


print A

CSC = A.to_csc()
CSR = A.to_csr()


A_i, A_j, A_v = A.find()
CSR_i, CSR_j, CSR_v = CSR.find()
CSC_i, CSC_j, CSC_v = CSC.find()

print "=" * 80

print A_i
print CSR_i
print CSC_i
print

print A_j
print CSR_j
print CSC_j
print

print A_v
print CSR_v
print CSC_v

print "$" * 80


print CSR[CSR_i[0], CSR_j[0]]

print CSC[CSC_i[0], CSC_j[0]]
