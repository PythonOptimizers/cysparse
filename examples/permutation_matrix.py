from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

P_mat_nd = np.array([0, 2, 1], dtype=np.int32)

P = NewPermutationLLSparseMatrix(P=P_mat_nd, size=3, itype=types.INT32_T, dtype=types.COMPLEX64_T)

print P

