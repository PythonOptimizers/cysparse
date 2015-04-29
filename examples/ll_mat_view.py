from sparse_lib.sparse.ll_mat import LLSparseMatrix, MakeLLSparseMatrix
import sys

A = MakeLLSparseMatrix(nrow=2, ncol=4, size_hint=10)
A.put_triplet([0,1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 1, 1, 1 ,1], [0, 1, 2, 3, 0 , 1, 2 , 3])

A.print_to(sys.stdout)

A_view = A[0:2, 0:4:2]
A_view_copy = A_view.matrix_copy()
A_view_copy.print_to(sys.stdout)

A_view2 = A_view[1, 0:2]
print "$" * 50
print A_view2.nnz
print "/" * 50
A_view2_copy = A_view2.matrix_copy()
A_view2_copy.print_to(sys.stdout)
print "!" * 50
A_view3 = A_view[::-1, ::-1]
A_view3_copy = A_view3.matrix_copy()
A_view3_copy.print_to(sys.stdout)