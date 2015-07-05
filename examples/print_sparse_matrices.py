from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLinearFillLLSparseMatrix(nrow=12, ncol=6, size_hint=40, store_zeros=True)
l1.clear_submatrix(1,3, 4,6)
print l1

print "*" * 80

l1.print_to(sys.stdout)

print "v" * 80

print l1.T