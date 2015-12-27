from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

import sys

A = NewLinearFillLLSparseMatrix(nrow=5, ncol=34)

print A

A_CSC = A.to_csc()

A_CSR = A_CSC.to_csr()

print A_CSR.are_column_indices_sorted()

print "*" * 80

A_CSR = A.to_csr()
A_CSC = A_CSR.to_csc()

print A_CSC.are_row_indices_sorted()