from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

from cysparse.linalg.spqr_context import NewSPQRContext


A = NewLinearFillLLSparseMatrix(size=4, is_symmetric=True)

print A

AA = A.to_ndarray()

print "WWW"
print np.linalg.det(AA)



A.to_csc().debug_print()

solver = NewSPQRContext(A)

print solver

solver.create_symbolic(ordering=1)
print solver.spqr_statistics()

solver.create_numeric()

solver.factorize(ordering=4)
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
b = np.ones(4, dtype=np.float64)
#
sol = solver.solve_expert(b, 'SPQR_RETX_EQUALS_B')

sol1 = np.linalg.solve(AA, b)

print sol1
#
print sol

print "3" * 80

print np.dot(AA,sol1)
print A * sol