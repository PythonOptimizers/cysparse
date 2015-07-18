from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

A = NewLinearFillLLSparseMatrix(nrow=5, ncol=5, store_zeros=True, is_symmetric=False)

print A


print "hooho"
try:
    from cysparse.utils.spy import spy
    print "he"
    import pylab
    import matplotlib.pyplot as pylab

    spy(A)
    pylab.show()
    print "hrllo"
except:
    pass

print "hrllodssgs"