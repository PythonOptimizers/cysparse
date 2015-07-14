from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

#A = NewLinearFillLLSparseMatrix(nrow=3, ncol=3, itype=types.INT64_T, dtype=types.FLOAT64_T)

#print A[:, :]
A = NewLLSparseMatrix(size=3, itype=types.INT64_T, dtype=types.FLOAT64_T)


B= np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.float64)
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

b = np.ones(3, dtype=np.float64)

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

