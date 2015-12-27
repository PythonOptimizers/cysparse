from cysparse.sparse.ll_mat import NewBandLLSparseMatrix

import numpy as np

diag = np.array([1, 2, 3], dtype=np.float64)

A = NewBandLLSparseMatrix(diag_coeff=[0], numpy_arrays=[diag], size=3)

print A
