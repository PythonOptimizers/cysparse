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

l1[2, 2] = 450000000000000000000  # huge number
l1[9, 9] = np.inf
l1[0, 0] = np.nan

l1.put_triplet([1,1], [1, 2], [5.6, 6.7])  # i, j, val
print l1

print l1[2, 2]
print l1[0, 0]             # was not assigned -> 0.0 by default

# like a dict
print l1.keys()            # (i, j)
print l1.values()          # val
print l1.items()           # ((i,j), val)

# returns 3 NumPy arrays with **corresponding** types!
print l1.find()

l1[4, 5] = 98374983.093843483
l1.clear_submatrix(1,3,4,7)

l1[2,0] = 5
l1[1,0] = 8
l1[0,0] = 4
l1.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

CSR = l1.to_csr()

CSR.print_to(sys.stdout)

print CSR.are_column_indices_sorted()