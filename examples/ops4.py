from cysparse.sparse.ll_mat import *

A = LinearFillLLSparseMatrix(nrow=4, ncol=3, dtype=FLOAT64_T).to_csc()#, first_element=1-5j, step=2+4j)
print A

B = LinearFillLLSparseMatrix(nrow=4, ncol=3, dtype=FLOAT64_T)
print B

C = LinearFillLLSparseMatrix(nrow=3, ncol=4, dtype=FLOAT64_T)
print B

import numpy as np

b = np.ones(4, dtype=np.float64)

D = 2 * A * 2 + A

print D.to_ll()

E = 5 * A

print E.to_ll()

print "=" * 80

F = (2 * A - B) * C

print F.to_ll()

print (A * C).to_ll()

print "=" * 80

G = (3 * A - 2 * A * 1) * A.T

print G.to_ll()

print (A * A.T).to_ll()