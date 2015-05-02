from cysparse.sparse.ll_mat import LLSparseMatrix, MakeLLSparseMatrix

# class LLPySparseMatrix(LLCySparseMatrix):
#  pass


matrix = MakeLLSparseMatrix(nrow=5, ncol=5, size_hint=1)
print matrix
matrix[1, 1] = 2
print matrix[1, 1]
print matrix[0, 4]
print matrix[2, 2]

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

matrix[1, 1] = 9
matrix[2, 2] = 10
matrix[3, 3] = 11

print matrix.nnz

matrix[1, 4] = 24323
matrix[2, 4] = -876387263872

print matrix.nnz

print '=' * 80
print "element: " + str(matrix[4, 4])

print matrix[4, 3]

ll_mat = MakeLLSparseMatrix(nrow=3, ncol=3, size_hint=4)
print ll_mat

import numpy as np

np_array = np.arange(24, dtype=np.dtype('d')).reshape((4, 6))
ll_mat2 = MakeLLSparseMatrix(matrix=np_array)

print ll_mat2

print "?" * 80

ll_mat2[0, 0] = 3453.34098309384039840934

import sys

ll_mat2.print_to(sys.stdout)

print '&' * 80
csr_mat = ll_mat2.to_csr()


print csr_mat

csr_mat.print_to(sys.stdout)

csr_mat.set_col(3, 32)
csr_mat.set_col(2, 6)
csr_mat.set_col(1, 99)

csr_mat.set_col(3, 32)
csr_mat.set_col(6, 32)

csr_mat.set_col(19, 32)

#csr_mat.set_col(3, 32)
#csr_mat.set_col(3, 32)
#csr_mat.set_col(3, 32)
#csr_mat.set_col(3, 32)

if csr_mat.are_column_indices_sorted():
    print "column indices are sorted!!!"


csr_mat.debug_print()


csr_mat.order_column_indices()


csr_mat.debug_print()

print "shape is "
print csr_mat.shape




