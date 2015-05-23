from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

l1 = NewLLSparseMatrix(nrow=3, ncol=3, size_hint=4)
print l1
print type(l1)

l1.put_triplet([1,1], [1, 2], [5.6, 6.7])
print l1

a = np.array([1,1,1], dtype=np.float64)

c = l1 * a

print c

print "+" * 80

l2 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=9, dtype=types.COMPLEX128_T, itype=types.INT64_T)

l2.put_triplet([1,1], [1,2], [34.77+99j, 43.4657])

b = np.array([1+1j, 1+1j, 1+1j, 1+1j], dtype=np.complex128)

c = l2 * b

print c


