from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=10, ncol=10, size_hint=40)
print l1
print type(l1)             # lots of classes are used internally...

# all memories given in bits
print l1.memory_element()  # memory for one element
print l1.memory_virtual()  # memory needed if a dense matrix was used
print l1.memory_real()     # memory used internally for the C-arrays

l1.compress()              # shrink the matrix as much as possible
print l1.memory_real()


for i in xrange(10):
    l1[1, i] = 1

l1.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

v = np.arange(0, 10, dtype=np.float64)

l1.col_scale(v)

l1.print_to(sys.stdout)