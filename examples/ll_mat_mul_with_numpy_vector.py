from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np
import sys

l1 = NewLLSparseMatrix(nrow=3, ncol=3, size_hint=4, __is_symmetric=True)
print l1
print type(l1)

l1.put_triplet([2,1], [1, 1], [5.6, 6.7])
l1.print_to(sys.stdout)

l1_csc = l1.to_csc()
l1_csc.print_to(sys.stdout)


a = np.array([1,1,1], dtype=np.float64)

c = l1 * a
z = l1_csc.matvec(a)

print "%" * 80
print c
print z

print "+" * 80

l2 = NewLLSparseMatrix(nrow=3, ncol=4, size_hint=9, dtype=types.COMPLEX128_T, itype=types.INT64_T)

l2.put_triplet([1,1], [1,2], [34.77+99j, 43.4657])

l2_csc = l2.to_csc()

b = np.array([1+1j, 1+1j, 1+1j, 1+1j], dtype=np.complex128)

c = l2 * b
z = l2_csc.matvec(b)

print c
print z

print "=" * 80

d = np.array([1, 1, 1], dtype=np.float64)

Ad = l1 * d

print Ad

Atd = l1.matvec_transp(d)

print Atd




