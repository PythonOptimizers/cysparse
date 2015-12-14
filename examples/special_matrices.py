from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

m = 5

n = 6

A = NewUnityLLSparseMatrix(nrow=m, ncol=n)
A.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

B = NewDiagonalLLSparseMatrix(nrow=m, ncol=n, element=9, dtype=types.COMPLEX64_T)
B.print_to(sys.stdout)

