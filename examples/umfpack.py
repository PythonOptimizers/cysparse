from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

A = NewLinearFillLLSparseMatrix(nrow=4, ncol=4, itype=types.INT64_T, dtype=types.COMPLEX128_T)

A.put_triplet([0, 0], [0, 1], [3, 5.6])

print A

from cysparse.solvers.umfpack import NewUmfpackSolver

solver = NewUmfpackSolver(A)

solver.create_numeric()

solver.report_numeric()

print "*" * 80

b = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128)

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
