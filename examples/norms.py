from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=4, ncol=4, size_hint=10, is_symmetric=True)
print l1
print type(l1)             # lots of classes are used internally...

# all memories given in bits
print l1.memory_element()  # memory for one element
print l1.memory_virtual()  # memory needed if a dense matrix was used
print l1.memory_real()     # memory used internally for the C-arrays

l1.compress()              # shrink the matrix as much as possible
print l1.memory_real()

l1[2, 2] = 4


l1.put_triplet([1,2], [1, 1], [5.6, 6.7])  # i, j, val
for i in range(4):
    l1[i, 0] = i *2 / 3 + 1.5

l1[1, 1] = np.inf
l1[2, 2] = np.nan
l1[3, 3] = 2.7

l1.print_to(sys.stdout)

########################################################################################################################
print "=" * 80

print "Symmetric matrix:"
print "inf norm = " + str(l1.norm('inf'))
print "1 norm = " + str(l1.norm('1'))
print "frob norm = " + str(l1.norm('frob'))
print '*' * 30
print "Non symmetric version:"
l1.generalize()
print "inf norm = " + str(l1.norm('inf'))
print "1 norm = " + str(l1.norm('1'))
print "frob norm = " + str(l1.norm('frob'))


########################################################################################################################
print "=" * 80

l2 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=10, is_symmetric=True)
l2.put_triplet([0, 1, 1, 2, 2], [0, 0, 1, 0, 1], [1, 2, 3, 4, 5]) # i, j, val

l2_norm_1_sym = l2.norm('1')
l2_norm_inf_sym = l2.norm('inf')
print l2_norm_1_sym
print l2_norm_inf_sym

l2.generalize()

l2_norm_1 = l2.norm('1')
l2_norm_inf = l2.norm('inf')
print l2_norm_1
print l2_norm_inf

assert l2_norm_1_sym == l2_norm_1
assert l2_norm_inf_sym == l2_norm_inf

########################################################################################################################
print "=" * 80
