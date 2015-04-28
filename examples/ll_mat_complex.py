from sparse_lib.sparse.ll_mat import LLSparseMatrix, MakeLLSparseMatrix

import sys

is_complex = True

matrix = MakeLLSparseMatrix(nrow=2, ncol=3, size_hint=10, is_complex=is_complex)
print matrix

if is_complex:
    matrix[0, 0] = complex(1, 45.68565)
    matrix[0, 2] = complex(3.6, -23.2342)
    matrix[1, 1] = complex(1)
    matrix[1, 2] = complex(-2)
else:
    matrix[0, 0] = 1
    matrix[0, 2] = 3.6
    matrix[1, 1] = 1
    matrix[1, 2] = -2

matrix.print_to(sys.stdout)

for elem in matrix.values():
    print elem

print "*" * 80

A = matrix.to_csr()

print A
A.print_to(sys.stdout)

matrix.save_to("togolo.mtx", "MM")