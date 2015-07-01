from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLinearFillLLSparseMatrix(nrow=10, ncol=10, size_hint=40)
print l1



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

print B.nnz
