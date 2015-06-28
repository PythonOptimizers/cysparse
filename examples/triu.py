from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

m = 5

n = 3

A = NewLinearFillLLSparseMatrix(nrow=m, ncol=n)
A.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

A.triu().print_to(sys.stdout)

########################################################################################################################
print "=" * 80
A.triu(include_diagonal=False).print_to(sys.stdout)

########################################################################################################################
print "=" * 80
