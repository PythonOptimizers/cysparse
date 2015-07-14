from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

A = NewLinearFillLLSparseMatrix(nrow=5, ncol=5, store_zeros=True, is_symmetric=True)

print A

C = A
print C

for i in xrange(0, -C.nrow, -1):
    print "** k = %d" % i
    D = C.tril(i)
    print "nnz = %d" % D.nnz
    print D

print "2" * 80

for i in xrange(C.ncol):
    print "** k = %d" % i
    D = C.triu(i)
    print "nnz = %d" % D.nnz
    print D


