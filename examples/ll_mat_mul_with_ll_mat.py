from cysparse.sparse.ll_mat import *
import cysparse.types.cysparse_types as types
import numpy as np

import sys

l1 = NewLLSparseMatrix(nrow=3, ncol=3, size_hint=4, dtype=types.INT32_T)

l1.put_triplet([1,1], [1, 2], [5, 6])

l1.print_to(sys.stdout)

c = l1 * l1

c.print_to(sys.stdout)

print "=" * 80

l2 = NewLLSparseMatrix(size=2, dtype=types.COMPLEX64_T)

l2.put_triplet([0, 0], [0, 1], [1+1j, 2+2j])

l2.print_to(sys.stdout)

d = l2 * l2

d.print_to(sys.stdout)

print "=" * 80

l3 = NewLLSparseMatrix(size=2, dtype=types.FLOAT32_T)

l3.put_triplet([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, np.inf])

l3.print_to(sys.stdout)

e = l3 * l3

e.print_to(sys.stdout)

print "=" * 80

A = NewLinearFillLLSparseMatrix(size=5)

I = NewDiagonalLLSparseMatrix(size=5, element=1)

print A * I
print I * A
print A

print "?" * 80

B = NewArrowheadLLSparseMatrix(size=3)

print B

print B * B



