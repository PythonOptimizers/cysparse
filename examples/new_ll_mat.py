from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

l1 = NewLLSparseMatrix(nrow=3, ncol=7, size_hint=400)
print l1

l2 = NewLLSparseMatrix(size=10, dtype=types.FLOAT32_T, itype=types.INT64_T)
print l2

l3 = NewLLSparseMatrix(size=4, itype=types.INT64_T, dtype=types.COMPLEX128_T, store_zeros=True)
print l3

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
l1.put_triplet([1,1], [1, 2], [5.6, 6.7])
print l1

print l1.keys()
print l1.values()
print l1.items()
print l1.find()

print l1[2, 2]
print l1[0, 0]

l2[2, 2] = 45000000

print l2[2, 2]
print l2[0, 0]
print l2.keys()
print l2.values()
print l2.items()
print l2.find()

l3[2,2] = 67.0+5.0j
l3[0,1] = 4.5-7.5j
print l3[0,0]
print l3[2, 2]
print l3
print l3.keys()
print l3.values()
print l3.items()
print l3.find()