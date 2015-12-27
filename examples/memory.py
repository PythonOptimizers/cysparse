from cysparse.sparse.ll_mat import *

A = ArrowheadLLSparseMatrix(nrow=50, ncol=800, itype=INT32_T, dtype=COMPLEX128_T)

print A

print "In bytes:"
print A.memory_real_in_bytes()
print A.memory_virtual_in_bytes()
print A.memory_element_in_bytes()

print "In bits:"
print A.memory_real_in_bits()
print A.memory_virtual_in_bits()
print A.memory_element_in_bits()

A.compress()
print A.memory_real_in_bytes()
print A.memory_real_in_bits()

print A

print "=" * 80

print A.memory_element_in_bits()
print A.memory_element_in_bytes()

print A.memory_index_in_bits()
print A.memory_index_in_bytes()