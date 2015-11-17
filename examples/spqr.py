from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

from cysparse.linalg.spqr_context import NewSPQRContext



A = NewLinearFillLLSparseMatrix(nrow=5, ncol=4)

print A

AA = A.to_ndarray()

print "WWW"
#print np.linalg.det(AA)



A.to_csc().debug_print()

solver = NewSPQRContext(A)

print solver

solver.create_symbolic(ordering='SPQR_ORDERING_NATURAL')
print solver.spqr_statistics()

solver.create_numeric()

solver.factorize(ordering='SPQR_ORDERING_CHOLMOD')
print solver.spqr_statistics()

print "$" * 80

# solver.analyze()
#
# print "Factor OK? " + str(solver.check_factor())
#
# solver.factorize()
#
# print "+" * 80
#
b = np.ones(5, dtype=np.float64)
#
sol = solver.solve(b)
sol_default = solver.solve_default(b)

#sol1 = np.linalg.solve(AA, b)

#print sol1
#
print sol
print sol_default

print "3" * 80

#print np.dot(AA,sol1)
print A * sol
print solver.SPQR_drop_tol_used()

print A * sol_default

print "%" * 80

print solver.SPQR_ordering_list()

print "D" * 80

Q, R, E = solver.get_QR(ordering='SPQR_ORDERING_BEST', econ=32)

print "Q:"
print Q.are_row_indices_sorted()
print Q
print Q.debug_print()

print "R:"
print R.are_row_indices_sorted()
print R
print R.debug_print()

print "%" * 90
print "well constructed?"
print Q.is_well_constructed(raise_exception=True)
print R.is_well_constructed()

print "CRASH" * 20
Q_np = Q.to_ndarray()
R_np = R.to_ndarray()


print Q
print Q_np

print "1" * 80
print R
print R_np

print np.dot(Q_np, R_np)

#print Q.to_ndarray() * R.to_ndarray()

print "Z" * 80
print E