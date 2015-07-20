from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys


A = NewLinearFillLLSparseMatrix(size=4, itype=types.INT64_T)

from cysparse.linalg.mumps_context import NewMumpsContext

context = NewMumpsContext(A)

print context.MUMPS_VERSION