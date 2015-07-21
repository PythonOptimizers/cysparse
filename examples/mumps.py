from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
from cysparse.linalg.mumps_context import NewMumpsContext

import numpy as np

import sys


A = NewLLSparseMatrix(mm_filename=sys.argv[1], itype=types.INT32_T, dtype=types.FLOAT64_T)

print A


context = NewMumpsContext(A, verbose=True)

print context.version_number

context.analyze()


context.factorize()