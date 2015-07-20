from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys


A = NewLinearFillLLSparseMatrix(size=4, itype=types.INT64_T, dtype=types.FLOAT64_T)

from cysparse.linalg.mumps_context import NewMumpsContext

context = NewMumpsContext(A)

print context.version_number

context.icntl[1] = 0
context.icntl[2] = 0
context.icntl[3] = 0
context.icntl[4] = 0

print context.icntl[1]

context.icntl[22] = 1
