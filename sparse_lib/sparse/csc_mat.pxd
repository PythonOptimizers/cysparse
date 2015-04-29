from sparse_lib.cysparse_types cimport *

from sparse_lib.sparse.sparse_mat cimport ImmutableSparseMatrix


cdef class CSCSparseMatrix(ImmutableSparseMatrix):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    cdef:
        FLOAT_t  *val;		 # pointer to array of values
        FLOAT_t  *ival    # pointer to array of imaginary values
        INT_t    *row;		 # pointer to array of indices
        INT_t    *ind;		 # pointer to array of indices

        bint __col_indices_sorted_test_done  # we only test this once
        bint __col_indices_sorted  # are the column indices sorted in ascending order?
        INT_t __first_row_not_ordered # first row that is not ordered

    cdef at(self, INT_t i, INT_t j)
    cdef safe_at(self, INT_t i, INT_t j)

cdef MakeCSCSparseMatrix(INT_t nrow, INT_t ncol, INT_t nnz, INT_t * ind, INT_t * row, FLOAT_t * val)
cdef MakeCSCComplexSparseMatrix(INT_t nrow, INT_t ncol, INT_t nnz, INT_t * ind, INT_t * row, FLOAT_t * val, FLOAT_t * ival)