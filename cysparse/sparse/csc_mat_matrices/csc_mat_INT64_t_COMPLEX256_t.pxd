from cysparse.common_types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT64_t_COMPLEX256_t cimport ImmutableSparseMatrix_INT64_t_COMPLEX256_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX256_t cimport LLSparseMatrix_INT64_t_COMPLEX256_t


cdef class CSCSparseMatrix_INT64_t_COMPLEX256_t(ImmutableSparseMatrix_INT64_t_COMPLEX256_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    cdef:
        COMPLEX256_t     *val		 # pointer to array of values
        INT64_t    *row		 # pointer to array of indices
        INT64_t    *ind		 # pointer to array of indices

        bint __row_indices_sorted_test_done  # we only test this once
        bint __row_indices_sorted  # are the column indices sorted in ascending order?
        INT64_t __first_col_not_ordered # first row that is not ordered


    cpdef bint is_well_constructed(self, bint raise_exception=?)  except False

    cdef _order_row_indices(self)
    cdef _set_row_indices_ordered_is_true(self)

    cdef INT64_t count_nnz_by_column(self, INT64_t column_number)

    cdef at(self, INT64_t i, INT64_t j)

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX256_t safe_at(self, INT64_t i, INT64_t j) except *


cdef MakeCSCSparseMatrix_INT64_t_COMPLEX256_t(INT64_t nrow,
                                        INT64_t ncol,
                                        INT64_t nnz,
                                        INT64_t * ind,
                                        INT64_t * row,
                                        COMPLEX256_t * val,
                                        bint use_symmetric_storage,
                                        bint use_zero_storage,
                                        bint row_indices_are_sorted=?)