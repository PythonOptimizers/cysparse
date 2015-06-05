from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT64_t_INT32_t cimport ImmutableSparseMatrix_INT64_t_INT32_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_INT32_t cimport LLSparseMatrix_INT64_t_INT32_t


cdef class CSCSparseMatrix_INT64_t_INT32_t(ImmutableSparseMatrix_INT64_t_INT32_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    cdef:
        INT32_t  *val;		 # pointer to array of values
        INT32_t  *ival    # pointer to array of imaginary values
        INT64_t    *row;		 # pointer to array of indices
        INT64_t    *ind;		 # pointer to array of indices

        bint __col_indices_sorted_test_done  # we only test this once
        bint __col_indices_sorted  # are the column indices sorted in ascending order?
        INT64_t __first_row_not_ordered # first row that is not ordered

    cdef at(self, INT64_t i, INT64_t j)

    cdef INT32_t safe_at(self, INT64_t i, INT64_t j) except? 2


cdef MakeCSCSparseMatrix_INT64_t_INT32_t(INT64_t nrow, INT64_t ncol, INT64_t nnz, INT64_t * ind, INT64_t * row, INT32_t * val)