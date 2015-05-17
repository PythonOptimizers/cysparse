from cysparse.sparse.ll_mat import MakeLLSparseMatrix
from cysparse.sparse.csc_mat import CSCSparseMatrix
from cysparse.sparse.csr_mat import CSRSparseMatrix

import sys

A = MakeLLSparseMatrix(size=4)

A.put_triplet([1.0, 2.0, 6.0, 3.0, 5.0, 4.0], [0, 1, 1, 2, 3, 3], [0, 1, 2, 2, 0, 3])

A.print_to(sys.stdout)

R = A.to_csr()

R.print_to(sys.stdout)

C = A.to_csc()

C.debug_print()
C.print_to(sys.stdout)

print "*" * 80

RC = R * C
print RC
RC.print_to(sys.stdout)