from cysparse.sparse.ll_mat import *
import cysparse.cysparse_types.cysparse_types as types
import numpy as np

import sys

from cysparse.linalg.cholmod_context import NewCholmodContext, cholmod_detailed_version, cholmod_version


A = NewLinearFillLLSparseMatrix(size=4, is_symmetric=True)

print A

A.to_csc().debug_print()

cholmod = NewCholmodContext(A)

print cholmod

print cholmod.CHOLMOD_VERSION

print cholmod_version()
print cholmod_detailed_version()

cholmod.request_GPU()
print cholmod.c_print
cholmod.c_print = 4
print cholmod.c_print

print "Checking if internal matrix is OK: " + str(cholmod.check_matrix())

cholmod.print_common_struct()

cholmod.print_sparse_matrix()

print "$" * 80

cholmod.analyze()

print "Factor OK? " + str(cholmod.check_factor())

cholmod.factorize()

print "+" * 80

b = np.ones(4, dtype=np.float64)

sol = cholmod.solve(b)

print sol