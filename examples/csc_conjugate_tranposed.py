from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX64_T)

l1.put_triplet([1, 1, 1], [0, 1, 2], [1+1j, 1-1.0j, 2+1j])  # i, j, val

l1.print_to(sys.stdout)

b = np.array([1+1j, 1+1j], dtype=np.complex64)

print b
########################################################################################################################
print "=" * 80

csc = l1.to_csc()

csc.print_to(sys.stdout)

y = csc.matvec_htransp(b)
y_bis = l1.matvec_htransp(b)

print y
print y_bis

x = csc.matvec_transp(b)
x_bis = l1.matvec_transp(b)

print
print x
print x_bis

########################################################################################################################
print "=" * 80

print "Symmetric case (non C-contiguous)"

l2 = NewLLSparseMatrix(size=3, size_hint=6, dtype=COMPLEX64_T, is_symmetric=True)
l2.put_triplet([0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1 ,2], [0, 1, 2, 3, 4, 5])

l2.print_to(sys.stdout)

c = np.array([1+1j, 1+1j, -2j], dtype=np.complex64)

sym_csc = l2.to_csc()

sym_csc.print_to(sys.stdout)

print sym_csc.matvec_htransp(c)
print l2.matvec_htransp(c)
