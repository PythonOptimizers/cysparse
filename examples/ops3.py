from cysparse.sparse.ll_mat import *
from cysparse.sparse.operator_proxies.sum_proxy import SumProxy
from cysparse.sparse.operator_proxies.mul_proxy import MulProxy
from cysparse.sparse.operator_proxies.scalar_mul_proxy import ScalarMulProxy

A = LinearFillLLSparseMatrix(nrow=4, ncol=3, dtype=FLOAT64_T)#, first_element=1-5j, step=2+4j)
print A

import numpy as np

b = np.ones(4, dtype=np.float64)

proxy = ScalarMulProxy(2, A)

print proxy.to_ll()

proxy2 = SumProxy(A, proxy)

print proxy2.to_ll()

proxy3 = MulProxy(proxy2, A.T)

print proxy3.to_ll()

print proxy3 * b