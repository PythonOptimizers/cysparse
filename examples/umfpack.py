from sparse_lib.sparse.ll_mat import MakeLLSparseMatrix
import sparse_lib.solvers.suitesparse.umfpack as umfpack

import sys

import numpy

A = MakeLLSparseMatrix(size=4)

A.put_triplet([1.0, 2.0, 6.0, 3.0, 5.0, 4.0], [0, 1, 1, 2, 3, 3], [0, 1, 2, 2, 0, 3])

solver = umfpack.UmfpackSolver(A)

print solver.UMFPACK_VERSION

solver.create_symbolic()
solver.create_numeric()

solver.report_control()

solver.report_info()

solver.report_symbolic()

solver.report_numeric()

print "*" * 80


b = numpy.array([1.0, 1.0, 1.0, 1.0])

sol = solver.solve(b)

print "Solution: ",
print sol
print "b: ",
print b
print "A*sol: ",
print A * sol

print "&" * 80

print solver.get_lunz()

(L, U, P, Q, D, do_recip, R) = solver.get_LU()

print L
L.print_to(sys.stdout)
print U
U.print_to(sys.stdout)
print P
print Q
print D
print do_recip
print R

print "*" * 80
print "L * U: "
LU = L * U


print "heheh" * 20
print LU
import sys
LU.print_to(sys.stdout)

print LU.memory_virtual()
print LU.memory_element()
print LU.memory_real()

LU.compress()

print "new matrix:"

print LU.memory_real()
LU.print_to(sys.stdout)