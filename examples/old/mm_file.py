#!/usr/bin/env python

from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types

import sys





print '=' * 80

l2 = LLSparseMatrixFromMMFile('zenios.mtx')
#l2.print_to(sys.stdout)
print l2
l2.debug_print()
print l2.are_column_indices_sorted()

print '+' * 80

l3 = LLSparseMatrix(mm_filename='conf5.0-00l4x4-1000.mtx.txt', mm_experimental=False, dtype=types.COMPLEX128_T)
print l3.dtype_str()
print l3

print "&" * 80
l3.debug_print()

print "$" * 80
print l3.are_column_indices_sorted()

#print l3
