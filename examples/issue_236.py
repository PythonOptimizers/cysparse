from cysparse.sparse.ll_mat import *

A = LLSparseMatrix(nrow=3, ncol=3)

A.put_triplet([0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0])
print A

from pysparse.sparse import PysparseMatrix as psp

J = psp(nrow=3, ncol=3)
J[0,0] = 1
J[1,1] = 2
J[2,2] = 3

print J

# Scale two first rows:"
J[:2,:] *= -1.0

print J

D = BandLLSparseMatrix(size=3, diag_coeff=[0], numpy_arrays=[np.array([-1, -1, 1], dtype=np.float64)])

A = A * D

print A.to_ll()