#!/usr/bin/env python

from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types

import sys

# all arguments **must** be named, this is not a bug but a feature (that can be changed)
# but I **strongly** recommend to keep it
# and to use shorter aliases without names...
A = NewLLSparseMatrix(mm_filename='simple_matrix.mm', itype=types.INT32_T, dtype=types.FLOAT64_T)

# if you don't know in advance what type of matrix is contained in your MM file, try
#B = NewLLSparseMatrixFromMMFile(filename='togolo.mtx')

print A
A.print_to(sys.stdout)

print '=' * 80

A_CSC = A.to_csc()
print A_CSC
A_CSC.print_to(sys.stdout)

print 'Internal arrays as NumPy arrays:'
ind, row, val = A_CSC.get_numpy_arrays()
print ind
print row
print val

print 'Internal arrays as C-pointers: (but you can not do anything with it in Python)'
# A_CSC.get_c_pointers()

print 'Diagonal as NumPy array:'
diag = A_CSC.diag()
print diag

########################################################################################################################
print "+" * 80
A_CSR = A.to_csr()
print A_CSR
A_CSR.print_to(sys.stdout)

print 'Internal arrays as NumPy arrays:'
ind, col, val = A_CSR.get_numpy_arrays()
print ind
print col
print val

print 'Diagonal as NumPy array:'
diag = A_CSR.diag()
print diag