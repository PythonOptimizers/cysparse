from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

import sys

m = 5

n = 3

print "Initial matrix"
A = NewLinearFillLLSparseMatrix(nrow=m, ncol=n, row_wise=False)
A.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

A.tril().print_to(sys.stdout)

########################################################################################################################
print "=" * 80
A.tril(k=-1).print_to(sys.stdout)

########################################################################################################################
print "=" * 80
print "+" * 80

B = NewLinearFillLLSparseMatrix(size=m, is_symmetric=True, row_wise=False)
B.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

B.tril().print_to(sys.stdout)

