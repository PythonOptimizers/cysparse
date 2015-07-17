from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT64_t cimport LLSparseMatrix_INT32_t_FLOAT64_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_FLOAT64_t cimport CSCSparseMatrix_INT32_t_FLOAT64_t

cdef class CholmodContext_INT32_t_FLOAT64_t:
    cdef:
        LLSparseMatrix_INT32_t_FLOAT64_t A

        INT32_t nrow
        INT32_t ncol
        INT32_t nnz

        # Matrix A in CSC format
        CSCSparseMatrix_INT32_t_FLOAT64_t csc_mat