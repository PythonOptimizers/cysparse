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


