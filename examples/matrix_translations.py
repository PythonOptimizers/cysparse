

from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys


A = NewLinearFillLLSparseMatrix(nrow=3, ncol=4, dtype=types.COMPLEX256_T, first_element=2-9.7j, step=-0.7j, row_wise=True)
A.clear_submatrix(2, 3, 1, 2)

print A

A_csr = A.to_csr()

print A_csr

print A_csr.to_csc()