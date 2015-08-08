from cysparse.sparse.ll_mat import *
import numpy as np


A = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=6)
A.put_triplet(np.array([0, 0, 1, 1, 1], dtype=np.int32), np.array([0, 1, 0, 1, 2],dtype=np.int32), np.array([2, 1, 3, 4, 5], 'd'))

print A