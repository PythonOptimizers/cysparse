#!/usr/bin/env python

from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

import sys


try:
    l1 = NewLLSparseMatrix(size=4, dtype=types.INT32_T, test_bounds=True)
    l1[0, 0] = 33333333333333333333
    l1.print_to(sys.stdout)
except:
    print "Nope didn't work ..."


print '=' * 80

l2 = NewLLSparseMatrixFromMMFile('togolo.mtx')
l2.print_to(sys.stdout)