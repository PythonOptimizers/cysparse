from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

m = 6

n = 9

A = NewLinearFillLLSparseMatrix(nrow=m, ncol=n, dtype=FLOAT32_T)
A.print_to(sys.stdout)

print "Main diagonal:"
print A.diag()

print
print "Positive diagonals:"
for k in xrange(1, n):
    print k, ': ', A.diag(k)

print
print "Negative diagonals:"
for k in xrange(-1, -m, -1):
    print k, ': ', A.diag(k)

print "*" * 80
print A.diags(slice(2, 5, 2))

