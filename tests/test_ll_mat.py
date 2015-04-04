from sparse_lib.sparse.ll_mat import LLSparseMatrix, make_ll_sparse_matrix

#class LLPySparseMatrix(LLCySparseMatrix):
#  pass


matrix = LLSparseMatrix(5, 5, 1)
print matrix
matrix[1,1] = 2
print matrix[1,1]
print matrix[0,4]
print matrix[2,2]

print matrix.nnz

try:
  matrix[100, 1000, 1] = -45343
except IndexError as e:
  print "value of error: %s" % e

try:
  matrix[100, 1000] = -45343
except IndexError as e:
  print "value of error: %s" % e


try:
  matrix[4, 4] = -45343
except IndexError as e:
  print "value of error: %s" % e


print matrix.nnz

matrix[1,1] = 9
matrix[2,2] = 10
matrix[3,3] = 11

print matrix.nnz

matrix[1,4] = 24323
matrix[2,4] = -876387263872

print matrix.nnz

print '=' * 80
print "element: " + str(matrix[4, 4])

print matrix[4, 3]

ll_mat = make_ll_sparse_matrix(nrow=3, ncol=3, size_hint=4)
print ll_mat


import numpy as np

np_array = np.arange(24, dtype=np.dtype('d')).reshape((4,6))
ll_mat2 = make_ll_sparse_matrix(matrix=np_array)

print ll_mat2



