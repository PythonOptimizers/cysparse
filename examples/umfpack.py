from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

A = NewLLSparseMatrix(nrow=3, ncol=5)

A.put_triplet([0, 0], [0, 1], [3, 5.6])

print A

from cysparse.solvers.umfpack import NewUmfpackSolver

solver = NewUmfpackSolver(A)

