from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=2, ncol=3, size_hint=10, dtype=COMPLEX64_T)

l1.put_triplet([1, 1, 1], [0, 1, 2], [1+1j, 1-1.0j, 2+1j])  # i, j, val

l1.print_to(sys.stdout)

########################################################################################################################
print "=" * 80
print "Proxies"
T = l1.T
print T

H = l1.H
print H

conj = l1.conj
print conj

########################################################################################################################
print "=" * 80
print "T for all proxies"
TT = T.T
print TT

HT = H.T
print HT

conjT = conj.T
print conjT

########################################################################################################################
print "=" * 80
print "H for all proxies"
TH = T.H
print TH

HH = H.H
print HH

conjH = conj.H
print conjH

########################################################################################################################
print "=" * 80
print "conj for all proxies"
Tconj = T.conj
print Tconj

Hconj = H.conj
print Hconj

conjconj = conj.conj
print conjconj