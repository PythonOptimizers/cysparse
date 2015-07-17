from pysparse.sparse import spmatrix
from pysparse.direct import umfpack
import numpy
n = 100
A = poisson2d_sym_blk(n)
b = numpy.ones(n*n)
x = numpy.empty(n*n)
LU = umfpack.factorize(A, strategy="UMFPACK_STRATEGY_SYMMETRIC")
LU.solve(b, x)
