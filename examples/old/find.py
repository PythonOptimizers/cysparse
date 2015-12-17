from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

A = NewLinearFillLLSparseMatrix(nrow=4, ncol=3, dtype=FLOAT32_T)

A_i, A_j, A_v = A.find()
CSR_i, CSR_j, CSR_v = A.to_csr().find()
CSC_i, CSC_j, CSC_v = A.to_csc().find()

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