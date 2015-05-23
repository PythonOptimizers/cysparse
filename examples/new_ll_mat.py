from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=10, ncol=10, size_hint=40)
print l1
print type(l1)             # lots of classes are used internally...

# all memories given in bits
print l1.memory_element()  # memory for one element
print l1.memory_virtual()  # memory needed if a dense matrix was used
print l1.memory_real()     # memory used internally for the C-arrays

l1.compress()              # shrink the matrix as much as possible
print l1.memory_real()

l1[2, 2] = 450000000000000000000  # huge number
l1[9, 9] = np.inf
l1[0, 0] = np.nan

l1.put_triplet([1,1], [1, 2], [5.6, 6.7])  # i, j, val
print l1

print l1[2, 2]
print l1[0, 0]             # was not assigned -> 0.0 by default

# like a dict
print l1.keys()            # (i, j)
print l1.values()          # val
print l1.items()           # ((i,j), val)

# returns 3 NumPy arrays with **corresponding** types!
print l1.find()

l1[4, 5] = 98374983.093843483

l1.print_to(sys.stdout)

########################################################################################################################
print "=" * 80
l2 = NewLLSparseMatrix(size=10, dtype=types.INT32_T, itype=types.INT32_T)
print l2

try:
    l2[2, 2] = 45000000000000000000000
except OverflowError:
    print "Value way to big to be stored in such matrix..."
    l2[2, 2] = 45000000

l2[5, 6] = 3.9              # we take the integer part, i.e. 3

print l2[2, 2]
print l2[0, 0]              # was not assigned -> 0 by default

print l2.keys()
print l2.values()
print l2.items()

print l2.find()
l2.print_to(sys.stdout)
########################################################################################################################
print "=" * 80
l3 = NewLLSparseMatrix(size=4, itype=types.INT64_T, dtype=types.COMPLEX128_T, store_zeros=True)
print l3

l3[2,2] = 67.0+5.0j
l3[0,1] = 4.5-7.5j
l3[3,2] = 0+0j               # zero **is** stored
print l3[0,0]                # not assigned -> 0+0j by default
print l3[2, 2]

l3.put_triplet([1,1], [1, 2], [5.6+1j, 6.7-7.8j])  # i, j, val
l3[0,0] = np.nan + np.nan * 1j
print l3


print "This is not a number: " + str(l3[0,0])

print l3.keys()
print l3.values()
print l3.items()

print l3.find()
l3.print_to(sys.stdout)
########################################################################################################################
print "=" * 80

l3_view = l3[0:2, 0]
print l3_view

l3_view_view = l3_view[0:1, 0]
print l3_view_view
print l3_view_view.is_empty

########################################################################################################################
print "=" * 80
l3_bis = l3_view_view.matrix_copy()
print l3_bis