from cysparse.sparse.ll_mat import *

nrow = 3
ncol = 4
nbr_elements = 4

A = LLSparseMatrix(nrow=nrow, ncol=ncol, size_hint=nbr_elements, itype=INT32_T, dtype=FLOAT64_T)

A.put_triplet([0,0,1,1], [1,2,0,2], [1.0, 2.0, 3.0, 4.0])
print A

A.debug_print()

print "T" * 80

CSC = A.to_csc()

print CSC

print '%' * 80

A_bis = CSC.to_ll()
#A_bis.debug_print()

print A_bis

A_bis.debug_print()

A_bis.compress()

print A_bis
#print "=" * 80
#CSR = A.to_csr()
#CSR.debug_print()

#print "8" * 49
#print CSR

#print A_bis