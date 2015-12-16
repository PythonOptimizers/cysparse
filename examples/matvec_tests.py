from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types
import numpy as np

import sys


A = NewLinearFillLLSparseMatrix(nrow=3, ncol=4, dtype=types.COMPLEX256_T, first_element=2-9.7j, step=-0.7j, row_wise=False)

A.print_to(sys.stdout)

a = np.array([1-0.4j,1,1], dtype=np.complex256)
b = np.array([-9.8j, 1.2+1.j, 0, -2], dtype=np.complex256)

def are_equal(A, B):

    if A.nrow == B.nrow and A.ncol == B.ncol:
        for i in xrange(A.nrow):
            for j in xrange(A.ncol):
                if A[i, j] != B[i, j]:
                    return False
    else:
        return False

    return True

########################################################################################################################
print "*" * 80
print "matvec"

c = A.matvec(b)
#d = A.H.matvec(b)
e = A.to_csc().matvec(b)
#f = A.to_csc().H.matvec(a)
g = A.to_csr().matvec(b)
#h = A.to_csr().H.matvec(a)

print c
#print d
print e
#print f
print g
#print h

########################################################################################################################
print "*" * 80
print "matvec_transp"

c = A.matvec_transp(a)
d = A.T.matvec(a)
e = A.to_csc().matvec_transp(a)
f = A.to_csc().T.matvec(a)
g = A.to_csr().matvec_transp(a)
h = A.to_csr().T.matvec(a)

print c
print d
print e
print f
print g
print h

########################################################################################################################
print "*" * 80
print "matvec_htransp"

c = A.matvec_htransp(a)
d = A.H.matvec(a)
e = A.to_csc().matvec_htransp(a)
f = A.to_csc().H.matvec(a)
g = A.to_csr().matvec_htransp(a)
h = A.to_csr().H.matvec(a)

print c
print d
print e
print f
print g
print h

########################################################################################################################
print "*" * 80
print "matvec_conj"

c = A.matvec_conj(b)
d = A.conj.matvec(b)
e = A.to_csc().matvec_conj(b)
f = A.to_csc().conj.matvec(b)
g = A.to_csr().matvec_conj(b)
h = A.to_csr().conj.matvec(b)

print c
print d
print e
print f
print g
print h
