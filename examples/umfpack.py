from sparse_lib.sparse.ll_mat import MakeLLSparseMatrix
import sparse_lib.solvers.suitesparse.umfpack as umfpack

matrix = MakeLLSparseMatrix(nrow=2, ncol=3, size_hint=10)
print matrix
matrix[0, 0] = 1
matrix[0, 2] = 3.6
matrix[1, 1] = 1
matrix[1, 2] = -2


solver = umfpack.UmfpackSolver(matrix)

print solver.UMFPACK_VERSION

solver.create_symbolic()
solver.create_numeric()

solver.report_control()

solver.report_info()

solver.report_symbolic()

solver.report_numeric()
