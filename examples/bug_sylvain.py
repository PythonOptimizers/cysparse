from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

import sys

#A = NewLLSparseMatrix(mm_filename='bcsstk01.mtx', itype=types.INT32_T, dtype=types.FLOAT64_T)
A = NewLLSparseMatrix(mm_filename='bug_sym_matrix.mtx', itype=types.INT32_T, dtype=types.FLOAT64_T)
#A = NewLLSparseMatrix(mm_filename='togolo.mtx', itype=types.INT32_T, dtype=types.FLOAT64_T)
(row, col, values) = A.find()
print row
print col
print values

A.print_to(sys.stdout)
print '=' * 80


# Converting A to compressed sparse column format
A_CSC = A.to_csc()
aind, arow, aval = A_CSC.get_numpy_arrays()
print aind
print arow
print aval


A_CSC.print_to(sys.stdout)

for i in xrange(A.nrow):
    for j in xrange(A.ncol):
        assert A[i, j] == A_CSC[i, j]