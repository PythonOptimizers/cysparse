

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
        double *val;		 # pointer to array of values
        int    *row;		 # pointer to array of indices
        int    *ind;		 # pointer to array of indices

        bint __status_ok     # do we have a completed CSR Matrix?
        bint __col_indices_sorted_test_done  # we only test this once
        bint __col_indices_sorted  # are the column indices sorted in ascending order?
        int __first_row_not_ordered # first row that is not ordered

cdef MakeCSCSparseMatrix(int nrow, int ncol, int nnz, int * ind, int * row, double * val)