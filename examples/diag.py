from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

m = 4

n = 9

A = NewLinearFillLLSparseMatrix(nrow=m, ncol=n, dtype=FLOAT32_T)
A.print_to(sys.stdout)

A_CSR = A.to_csr()
A_CSR.print_to(sys.stdout)

A_CSC = A.to_csr()
A_CSC.print_to(sys.stdout)

print "Main diagonal:"
print A.diag()
print A_CSR.diag()
print A_CSC.diag()

print
print "Positive diagonals:"
for k in xrange(1, n):
    print k, ': ', A.diag(k)
    print k, ': ', A_CSR.diag(k)
    print k, ': ', A_CSC.diag(k)

print
print "Negative diagonals:"
for k in xrange(-1, -m, -1):
    print k, ': ', A.diag(k)
    print k, ': ', A_CSR.diag(k)
    print k, ': ', A_CSC.diag(k)

print "*" * 80
print A.diags(slice(2, 5, 2))

