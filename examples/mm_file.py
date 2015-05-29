from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types

import sys

l1 = NewLLSparseMatrix(mm_filename='togolo.mtx', dtype=types.COMPLEX128_T)

l1.print_to(sys.stdout)
