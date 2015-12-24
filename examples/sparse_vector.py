from cysparse.sparse.ll_mat import *

v = LLSparseMatrix(nrow=4, ncol=1)
v.put_triplet([0, 2], [0, 0], [1.0, 2.0])

A = LinearFillLLSparseMatrix(nrow=3, ncol=4)

print v
print A

C = A * v
print C

A.T[0, 0] = 8.9