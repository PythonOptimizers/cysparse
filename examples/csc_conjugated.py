from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX64_T)

l1.put_triplet([1, 1, 1], [0, 1, 2], [1+1j, 1-1.0j, 2+1j])  # i, j, val

l1.print_to(sys.stdout)

b = np.array([1+1j, 1+1j, -2j], dtype=np.complex64)

print b
########################################################################################################################
print "=" * 80
print "test matvec_conj"
csc = l1.to_csc()

csc.print_to(sys.stdout)

y = csc.matvec_conj(b)
y_bis = l1.matvec_conj(b)

print y
print y_bis


########################################################################################################################
print "=" * 80
print "test matvec_transp"

c = np.array([1+1j, 1+1j], dtype=np.complex64)

print csc.matvec_transp(c)
print l1.matvec_transp(c)


########################################################################################################################
print "=" * 80
print "test matvec"

print csc.matvec(b)
print l1.matvec(b)

########################################################################################################################
print "=" * 80
print "test matvec_htransp"

print csc.matvec_htransp(c)
print l1.matvec_htransp(c)
