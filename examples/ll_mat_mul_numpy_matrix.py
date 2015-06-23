from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=6)
l1.put_triplet([0, 0, 1, 1, 1], [0, 1, 0, 1, 2], [2, 1, 3, 4, 5])

l1.print_to(sys.stdout)
print l1.nnz

n = np.array([[1, 2], [3, 4] , [5, 6]], dtype=np.float64)
print n
C = l1 * n

print C

