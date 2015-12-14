from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLinearFillLLSparseMatrix(ncol=10, nrow=10)

########################################################################################################################
print "=" * 80

print l1

l1.delete_cols([2, 3, 0])

print l1