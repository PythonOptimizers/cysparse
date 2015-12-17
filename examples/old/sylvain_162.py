import numpy as np
from cysparse.sparse.ll_mat import *

A = NewLLSparseMatrix(nrow=3, ncol=3, use_symmetric_storage=True)
A[:,:] = np.array([[1,2,3],[2,4,5],[3,5,6]])
#A[:,:] = np.array([[1,0,0],[2,4,0],[3,5,6]])
print A