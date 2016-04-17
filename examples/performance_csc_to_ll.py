from cysparse.sparse.ll_mat import *

from timeit import default_timer as timer

start = timer()
B = LinearFillLLSparseMatrix(nrow=3000, ncol=3000, first_element=1-6j, step=2+5j, dtype=COMPLEX64_T)
end = timer()
print "Creation of LL done in %1.1f second(s) [Poor result]" % (end - start)

start = timer()
B_CSC = B.to_csc()
end = timer()
print "Conversion LL -> CSC done in %1.1f second(s)" % (end - start)

start = timer()
B_bis = B_CSC.to_ll()
end = timer()
print "Conversion CSC -> LL done in %1.1f second(s)" % (end - start)
