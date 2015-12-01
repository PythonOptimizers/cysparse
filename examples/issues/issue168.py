from cysparse.sparse.ll_mat import *
from cysparse.cysparse_types.cysparse_types import *

A = NewLLSparseMatrix(mm_filename="zenios.mtx")

print A

print A.norm("frob")