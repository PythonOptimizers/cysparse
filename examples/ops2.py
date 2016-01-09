from cysparse.sparse.ll_mat import *
import numpy as np

A = LinearFillLLSparseMatrix(nrow=4, ncol=3, dtype=COMPLEX128_T, first_element=1-0.2j)
B = LinearFillLLSparseMatrix(nrow=3, ncol=4, dtype=COMPLEX128_T)
C = LinearFillLLSparseMatrix(size=4, dtype=COMPLEX128_T)

print A
print B
print C

v = np.ones(4, dtype=np.complex128)

print "=" * 80

proxy = A * A.T + C + C.conj + B.T * B

print proxy * v

proxy_ll = proxy.to_ll()

print proxy_ll

print
print proxy * v
print proxy_ll * v

print "+" * 80

A * 2