from sparse_lib.sparse.ll_mat import LLSparseMatrix, MakeLLSparseMatrix

# class LLPySparseMatrix(LLCySparseMatrix):
#  pass

import sys

matrix = MakeLLSparseMatrix(nrow=2, ncol=3, size_hint=10)
print matrix
matrix[0, 0] = 1
matrix[0, 2] = 3.6
matrix[1, 1] = 1
matrix[1, 2] = -2

matrix.print_to(sys.stdout)

matrix2 = MakeLLSparseMatrix(nrow=3, ncol=2, size_hint=10)

matrix2[0, 0] = 1
matrix2[1, 1] = 1

matrix2[2, 1] += 9

matrix2.print_to(sys.stdout)

print "=" * 80
C = matrix * matrix2

print C

import sys

C.print_to(sys.stdout)

print "$" * 80

import numpy as np
np_array = np.arange(3).astype(np.float64)

dd = matrix * np_array

print dd