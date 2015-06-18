from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=4, ncol=5, size_hint=10, dtype=COMPLEX64_T)

l1[2, 2] = 45000
l1.put_triplet([1,1], [1, 2], [5.6-1.0j, 6.7+3.2j])  # i, j, val

l1.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

H = l1.H

print H

HH = H.H

print HH

