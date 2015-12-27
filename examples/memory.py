from cysparse.sparse.ll_mat import *

A = ArrowheadLLSparseMatrix(nrow=50, ncol=800, itype=INT64_T, dtype=COMPLEX128_T)

print A

print A.memory_real()
print A.memory_virtual()
print A.memory_element()