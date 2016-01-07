from cysparse.sparse.ll_mat import *
from cysparse.sparse.operator_proxies.sum_proxy import SumProxy
from cysparse.sparse.operator_proxies.mul_proxy import MulProxy

A = LinearFillLLSparseMatrix(nrow=4, ncol=3, dtype=COMPLEX128_T)
print A

import numpy as np

b = np.ones(3, dtype=np.complex128)

proxy = SumProxy(A, A)

print proxy[0, 0]

print "proxy is A + A. What about to_ll()?"

proxy_LL = proxy.to_ll()

print proxy_LL

print proxy * b

print proxy_LL * b

C = proxy + A + proxy

print C[0,0]

print C * b

print "=" * 80

I = IdentityLLSparseMatrix(size=3, dtype=COMPLEX128_T)
print I

mul_proxy = MulProxy(A, I)

print mul_proxy[2,2]

print mul_proxy * b

print "=" * 80

B = LinearFillLLSparseMatrix(nrow=3, ncol=5, dtype=COMPLEX128_T)
H = SumProxy(A, A) * (SumProxy(B, B) + B + B)

print H[0,0]

o = np.ones(5, dtype=np.complex128)
print H * o

print H.to_ll()