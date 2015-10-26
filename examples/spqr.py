from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

from cysparse.linalg.spqr_context import NewSPQRContext


A = NewLinearFillLLSparseMatrix(size=4, is_symmetric=True)

print A

A.to_csc().debug_print()

solver = NewSPQRContext(A)

print solver




print "$" * 80

# solver.analyze()
#
# print "Factor OK? " + str(solver.check_factor())
#
# solver.factorize()
#
# print "+" * 80
#
# b = np.ones(4, dtype=np.float64)
#
# sol = solver.solve(b)
#
# print sol