from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT64_t_COMPLEX128_t cimport ImmutableSparseMatrix_INT64_t_COMPLEX128_t

cdef class CSRSparseMatrix_INT64_t_COMPLEX128_t(ImmutableSparseMatrix_INT64_t_COMPLEX128_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    cdef:
        COMPLEX128_t *  val		 # pointer to array of values
        INT64_t * col		 # pointer to array of indices
        INT64_t * ind		 # pointer to array of indices

        bint __col_indices_sorted_test_done  # we only test this once
        bint __col_indices_sorted            # are the column indices sorted in ascending order?
        INT64_t __first_row_not_ordered      # first row that is not ordered

cdef MakeCSRSparseMatrix_INT64_t_COMPLEX128_t(INT64_t nrow, INT64_t ncol, INT64_t nnz, INT64_t * ind, INT64_t * col, COMPLEX128_t * val)