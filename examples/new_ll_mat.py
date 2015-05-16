from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

l1 = NewLLSparseMatrix(nrow=3, ncol=7, size_hint=400)
print l1

l2 = NewLLSparseMatrix(size=10, dtype=types.INT32_T, itype=types.INT32_T)
#print l2

#l3 = NewLLSparseMatrix(size=4, is_complex=True, itype='INT64_t', is_symmetric=True, store_zeros=True)
#print l3

#l1.compress()  # first common method implemented! ;-)

print type(l1)

print "=" * 80
print l1.memory_element()
print l1.memory_virtual()
print l1.memory_real()
#print l2.memory_element()
#print l2.memory_virtual()
#print l3.memory_element()
#print l3.memory_virtual()

print "&" * 80

import cysparse.types.cysparse_types as types

print types.INT64_T

l1[2, 2] = 45000000000000000000000