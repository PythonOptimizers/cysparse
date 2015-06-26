from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

A = NewLinearFillLLSparseMatrix(nrow=3, ncol=4, dtype=types.COMPLEX256_T, row_wise=False)

A.print_to(sys.stdout)