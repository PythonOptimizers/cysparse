from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLinearFillLLSparseMatrix(nrow=10, ncol=10, size_hint=40)
print l1
print type(l1)             # lots of classes are used internally...



l1.print_to(sys.stdout)

A = l1[1:5, [2, 2, 2, 3]]

print A

print A.nnz