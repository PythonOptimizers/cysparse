from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_FLOAT64_t cimport ImmutableSparseMatrix_INT32_t_FLOAT64_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT64_t cimport LLSparseMatrix_INT32_t_FLOAT64_t


cdef class CSCSparseMatrix_INT32_t_FLOAT64_t(ImmutableSparseMatrix_INT32_t_FLOAT64_t):
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
        INT32_t    *row		 # pointer to array of indices
        INT32_t    *ind		 # pointer to array of indices

        bint __col_indices_sorted_test_done  # we only test this once
        bint __col_indices_sorted  # are the column indices sorted in ascending order?
        INT32_t __first_row_not_ordered # first row that is not ordered

    cdef at(self, INT32_t i, INT32_t j)

    cdef FLOAT64_t safe_at(self, INT32_t i, INT32_t j) except? 2


cdef MakeCSCSparseMatrix_INT32_t_FLOAT64_t(INT32_t nrow, INT32_t ncol, INT32_t nnz, INT32_t * ind, INT32_t * row, FLOAT64_t * val, bint is_symmetric, bint store_zeros)