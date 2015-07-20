from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX64_t cimport LLSparseMatrix_INT64_t_COMPLEX64_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX64_t cimport CSCSparseMatrix_INT64_t_COMPLEX64_t



cdef class MumpsContext_INT64_t_COMPLEX64_t:
    cdef:
        LLSparseMatrix_INT64_t_COMPLEX64_t A

        INT64_t nrow
        INT64_t ncol
        INT64_t nnz

        # Matrix A in CSC format
        CSCSparseMatrix_INT64_t_COMPLEX64_t csc_mat