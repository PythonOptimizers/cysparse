from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

import sys

nrow = 2
ncol = 3
A = LinearFillLLSparseMatrix(nrow=nrow, ncol=ncol, dtype=COMPLEX128_T, itype=INT32_T)

C = A[0:-1,0:-1]

print C

#for i in xrange(nrow):
#    for j in xrange(ncol):
#        print A[i, j]
#        C[i,j]

C[0, 0]
