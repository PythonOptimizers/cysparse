from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT64_t_FLOAT64_t cimport ImmutableSparseMatrix_INT64_t_FLOAT64_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT64_t cimport LLSparseMatrix_INT64_t_FLOAT64_t


cdef class CSCSparseMatrix_INT64_t_FLOAT64_t(ImmutableSparseMatrix_INT64_t_FLOAT64_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    cdef:
        FLOAT64_t     *val		 # pointer to array of values
        INT64_t    *row		 # pointer to array of indices
        INT64_t    *ind		 # pointer to array of indices

        bint __row_indices_sorted_test_done  # we only test this once
        bint __row_indices_sorted  # are the column indices sorted in ascending order?
        INT64_t __first_col_not_ordered # first row that is not ordered

    cdef at(self, INT64_t i, INT64_t j)

    cdef FLOAT64_t safe_at(self, INT64_t i, INT64_t j) except? 2


cdef MakeCSCSparseMatrix_INT64_t_FLOAT64_t(INT64_t nrow, INT64_t ncol, INT64_t nnz, INT64_t * ind, INT64_t * row, FLOAT64_t * val, bint is_symmetric, bint store_zeros)