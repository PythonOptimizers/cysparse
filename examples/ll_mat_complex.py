from sparse_lib.sparse.ll_mat import LLSparseMatrix, MakeLLSparseMatrix

import sys

matrix = MakeLLSparseMatrix(nrow=2, ncol=3, size_hint=10, is_complex=True)
print matrix
matrix[0, 0] = complex(1)
matrix[0, 2] = complex(3.6)
matrix[1, 1] = complex(1)
matrix[1, 2] = complex(-2)

matrix.print_to(sys.stdout)

for elem in matrix.values():
    print elem