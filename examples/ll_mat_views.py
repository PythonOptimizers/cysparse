from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLinearFillLLSparseMatrix(nrow=10, ncol=10, size_hint=40, dtype=types.COMPLEX256_T, store_zeros=True)
l1.clear_submatrix(1,3, 4,6)
print l1

l1[0, 0] = 9999999999999999996788989

l1[0, 1] = 0

#print l1.at_to_string(0, 0)
#print l1.at_to_string(10, 10)
#print l1.at_to_string(9, 9)
#print l1.at_to_string(2,6)

print "8" * 80

for i in xrange(10):
    for j in xrange(10):
        print l1.at_to_string(i, j), ' ',

    print

l1.print_to(sys.stdout)

A = l1[1:5, [2, 2, 2, 3]]

print A

print A.nnz

########################################################################################################################
print "*" * 80
l2 = NewLinearFillLLSparseMatrix(nrow=10, ncol=10, size_hint=40, is_symmetric=True)
print l2



l2.print_to(sys.stdout)

B = l1[1:5, [2, 2, 3]]

print B
print B.nnz

print "4" * 80
print B.get_matrix()

print  "9" * 80

C = NewLinearFillLLSparseMatrix(nrow=10, ncol=10, size_hint=40, dtype=types.COMPLEX256_T, store_zeros=True)
print C
print C[:3,:3]

print B[:3,:3]

#print C
print "before"
C[:3, :3] = B[:3,:3]
print "after"
print C