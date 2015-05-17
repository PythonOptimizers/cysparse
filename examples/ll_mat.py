from cysparse.sparse.ll_mat import LLSparseMatrix, MakeLLSparseMatrix

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
#np_array = np.arange(10).astype(np.float64)[::2]
dd = matrix * np_array

print dd

print "(" * 80

matrix_T = matrix.T

print matrix_T

matrix_T.print_to(sys.stdout)

np_array2 = np.arange(2).astype(np.float64)

res = matrix.T * np_array2

print res

print "*" * 80

ti = matrix[0:11:1, 0:3:3]

print type(ti)

print "!" * 89

l = [0, 1]

print l
print type(l)
 = matrix[l, 0:3]

print type()

print "8" * 80
matrix.compress()

print matrix[0,0]
print matrix[0, 2]

print "?" * 80

print matrix.keys()
print matrix.values()
print matrix.items()

print "=" * 80

matrix.print_to(sys.stdout)
matrix_c = MakeLLSparseMatrix(nrow=2, ncol=2, size_hint=10)
matrix_c[0, 0] = 1
matrix_c[0, 1] = 1
matrix_c[1, 0] = 1
matrix_c[1, 1] = 1
matrix[0:2, 1:3] = matrix_c

matrix.print_to(sys.stdout)