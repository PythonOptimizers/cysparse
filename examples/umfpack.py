from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

#A = NewLinearFillLLSparseMatrix(nrow=3, ncol=3, itype=types.INT64_T, dtype=types.FLOAT64_T)

itype = types.INT32_T
dtype = types.FLOAT64_T

np_dtype = np.float64

nrow = 3
ncol = 3
size = 3

#print A[:, :]
A = NewLLSparseMatrix(size=size, itype=itype, dtype=dtype)


B= np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np_dtype)
print B
print B[(0,2)]

A[0:3,0:3] = B

#A.put_triplet([0, 0], [0, 1], [3, 5.6])
#A[0, 0] = 0
print A

from cysparse.solvers.umfpack import NewUmfpackSolver

solver = NewUmfpackSolver(A)

solver.set_verbosity(0)

solver.create_numeric()

solver.report_numeric()

print "*" * 80

b = np.ones(3, dtype=np_dtype)

sol = solver.solve(b)

print "Solution: ",
print sol
print "b: ",
print b
print "A*sol: ",
print A * sol

print "&" * 80

print solver.get_lunz()

(L, U, P, Q, D, do_recip, R) = solver.get_LU()


print L
print U
print P
print Q
print D
print do_recip
print R

print "=" * 80

print L
print L.to_ndarray()

print U
print U.to_ndarray()

lhs = L * U

print "test L*U"
print lhs
print np.dot(L.to_ndarray(), U.to_ndarray())

#sys.exit(-1)

P_new = P.astype(np_dtype)

P_mat = NewBandLLSparseMatrix(diag_coeff=[0], numpy_arrays=[P_new], size=3, dtype=dtype, itype=itype)
print P_mat

Q_new = Q.astype(np_dtype)
Q_mat = NewBandLLSparseMatrix(diag_coeff=[0], numpy_arrays=[Q_new], size=3, dtype=dtype, itype=itype)
print Q_mat

if do_recip:
    R_mat = NewBandLLSparseMatrix(diag_coeff=[0], numpy_arrays=[R], size=3, dtype=dtype, itype=itype)
else:
    R_mat = NewLLSparseMatrix(size=3, dtype=dtype, itype=itype)
    for i in xrange(3):
        R_mat[i, i] = 1/R[i]
print R_mat

print "T" * 80
print "lhs = "
print lhs
rhs = P_mat * R_mat * A * Q_mat
print "rhs = "
print rhs