from cysparse.sparse.ll_mat import *
import cysparse.common_types.cysparse_types as types

A = NewLLSparseMatrix(mm_filename='../bcsstk01.mtx', itype=types.INT64_T, dtype=types.FLOAT64_T)
print A.is_symmetric
print A.to_ndarray()[0,:] # first line
print A.to_ndarray()[:,0] # first row
print A
print A.to_ndarray()