from sparse_lib.sparse.ll_mat import MakeLLSparseMatrix

import sys

A = MakeLLSparseMatrix(mm_filename="simple_matrix.mm")

print type(A)

A.print_to(sys.stdout)
