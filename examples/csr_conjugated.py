from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX64_T)

l1.put_triplet([1, 1, 1], [0, 1, 2], [1+1j, 1-1.0j, 2+1j])  # i, j, val

l1.print_to(sys.stdout)

b = np.array([1+1j, 1+1j, -2j], dtype=np.complex64)

print b
########################################################################################################################
print "=" * 80

csr = l1.to_csr()

csr.print_to(sys.stdout)

y = csr.matvec_conj(b)
y_bis = l1.matvec_conj(b)

print y
print y_bis

