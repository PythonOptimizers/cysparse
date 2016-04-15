#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
from cysparse.common_types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_FLOAT128_t cimport ImmutableSparseMatrix_INT32_t_FLOAT128_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT128_t cimport LLSparseMatrix_INT32_t_FLOAT128_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_FLOAT128_t cimport CSCSparseMatrix_INT32_t_FLOAT128_t

cdef class CSRSparseMatrix_INT32_t_FLOAT128_t(ImmutableSparseMatrix_INT32_t_FLOAT128_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    cdef:
        FLOAT128_t *  val		 # pointer to array of values
        INT32_t * col		 # pointer to array of indices
        INT32_t * ind		 # pointer to array of indices

        bint __col_indices_sorted_test_done  # we only test this once
        bint __col_indices_sorted            # are the column indices sorted in ascending order?
        INT32_t __first_row_not_ordered      # first row that is not ordered


    cdef _order_column_indices(self)
    cdef _set_column_indices_ordered_is_true(self)
    cdef at(self, INT32_t i, INT32_t j)
    # EXPLICIT TYPE TESTS

    cdef FLOAT128_t safe_at(self, INT32_t i, INT32_t j) except? 2


cdef MakeCSRSparseMatrix_INT32_t_FLOAT128_t(INT32_t nrow,
                                        INT32_t ncol,
                                        INT32_t nnz,
                                        INT32_t * ind,
                                        INT32_t * col,
                                        FLOAT128_t * val,
                                        bint store_symmetric,
                                        bint store_zero,
                                        bint col_indices_are_sorted=?)

cdef LLSparseMatrix_INT32_t_FLOAT128_t multiply_csr_mat_by_csc_mat_INT32_t_FLOAT128_t(CSRSparseMatrix_INT32_t_FLOAT128_t A, CSCSparseMatrix_INT32_t_FLOAT128_t B)