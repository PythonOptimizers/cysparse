from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

from cysparse.linalg.cholmod_context import NewCholmodContext, cholmod_detailed_version, cholmod_version


A = NewLinearFillLLSparseMatrix(size=4, is_symmetric=True)

cholmod = NewCholmodContext(A)

print cholmod

print cholmod.CHOLMOD_VERSION

print cholmod_version()
print cholmod_detailed_version()

cholmod.request_GPU()
print cholmod.c_print
cholmod.c_print = 4
print cholmod.c_print

cholmod.print_common_struct()

cholmod.print_sparse_matrix()