"""
Main entry to create ``LLSparseMatrix`` matrices.

Warning:
    ``LLSparseMatrix`` matrices can **only** be constructed through a *factory* function, **not** by direct
    instantiation.
"""

from cysparse.types.cysparse_types cimport *

#from cysparse.sparse.ll_mat_view cimport LLSparseMatrixView

cdef FLOAT32_t LL_MAT_INCREASE_FACTOR

cdef INT32_t LL_MAT_DEFAULT_SIZE_HINT

cpdef bint PyLLSparseMatrix_Check(object obj)

