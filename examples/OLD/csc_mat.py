from cysparse.sparse.ll_mat import MakeLLSparseMatrix

import sys

ll_mat = MakeLLSparseMatrix(nrow=3, ncol=4)
ll_mat[0, 0] = 3.4
ll_mat[0, 1] = 3.7
ll_mat[1, 0] = 2.8
ll_mat[2, 2] = 4
ll_mat[2, 3] = -10.4


ll_mat.print_to(sys.stdout)

csc_mat = ll_mat.to_csc()

csc_mat.debug_print()