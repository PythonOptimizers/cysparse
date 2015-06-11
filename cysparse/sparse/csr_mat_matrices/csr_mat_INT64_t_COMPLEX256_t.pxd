from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT64_t_COMPLEX256_t cimport ImmutableSparseMatrix_INT64_t_COMPLEX256_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX256_t cimport LLSparseMatrix_INT64_t_COMPLEX256_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX256_t cimport CSCSparseMatrix_INT64_t_COMPLEX256_t

cdef class CSRSparseMatrix_INT64_t_COMPLEX256_t(ImmutableSparseMatrix_INT64_t_COMPLEX256_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    cdef:
        COMPLEX256_t *  val		 # pointer to array of values
        INT64_t * col		 # pointer to array of indices
        INT64_t * ind		 # pointer to array of indices

        bint __col_indices_sorted_test_done  # we only test this once
        bint __col_indices_sorted            # are the column indices sorted in ascending order?
        INT64_t __first_row_not_ordered      # first row that is not ordered


    cdef _order_column_indices(self)
    cdef _set_column_indices_ordered_is_true(self)
    cdef at(self, INT64_t i, INT64_t j)
    # EXPLICIT TYPE TESTS

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX256_t safe_at(self, INT64_t i, INT64_t j) except *


cdef MakeCSRSparseMatrix_INT64_t_COMPLEX256_t(INT64_t nrow, INT64_t ncol, INT64_t nnz, INT64_t * ind, INT64_t * col, COMPLEX256_t * val, bint is_symmetric)

cdef LLSparseMatrix_INT64_t_COMPLEX256_t multiply_csr_mat_by_csc_mat_INT64_t_COMPLEX256_t(CSRSparseMatrix_INT64_t_COMPLEX256_t A, CSCSparseMatrix_INT64_t_COMPLEX256_t B)