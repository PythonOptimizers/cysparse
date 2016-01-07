from cysparse.sparse.ll_mat import *
from cysparse.sparse.operator_proxies.sum_proxy import SumProxy
from cysparse.sparse.operator_proxies.mul_proxy import MulProxy

A = LinearFillLLSparseMatrix(nrow=4, ncol=3)
print A

import numpy as np

b = np.ones(3)

proxy = SumProxy(A, A)

print proxy[0, 0]

print proxy * b

C = proxy + A + proxy

print C[0,0]

print C * b

print "=" * 80

I = IdentityLLSparseMatrix(size=3)
print I

mul_proxy = MulProxy(A, I)

print mul_proxy[2,2]

print mul_proxy * b


