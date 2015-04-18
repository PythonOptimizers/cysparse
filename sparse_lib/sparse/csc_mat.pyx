"""
Condensed Sparse Column (CSC) Format Matrices.


"""
from __future__ import print_function

from sparse_lib.sparse.sparse_mat cimport ImmutableSparseMatrix


from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class CSCSparseMatrix(ImmutableSparseMatrix):
    """
    Compressed Sparse Column Format matrix.

    Note:
        This matrix can **not** be modified.

    """


    def __cinit__(self, int nrow, int ncol, int nnz):
        self.__status_ok = False



    def __dealloc__(self):
        PyMem_Free(self.val)
        PyMem_Free(self.row)
        PyMem_Free(self.ind)


    ####################################################################################################################
    # DEBUG
    ####################################################################################################################
    def debug_print(self):
        cdef int i
        print("ind:")
        for i from 0 <= i < self.ncol + 1:
            print(self.ind[i], end=' ', sep=' ')
        print()

        print("row:")
        for i from 0 <= i < self.nnz:
            print(self.row[i], end=' ', sep=' ')
        print()

        print("val:")
        for i from 0 <= i < self.nnz:
            print(self.val[i], end=' == ', sep=' == ')
        print()

    def set_row(self, int i, int val):
        self.row[i] = val




########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSCSparseMatrix(int nrow, int ncol, int nnz, int * ind, int * row, double * val):
    """
    Construct a CSCSparseMatrix object.

    Args:
        nrow (int): Number of rows.
        ncol (int): Number of columns.
        nnz (int): Number of non-zeros.
        ind (int *): C-array with column indices pointers.
        row  (int *): C-array with row indices.
        val  (double *): C-array with values.
    """
    csc_mat = CSCSparseMatrix(nrow=nrow, ncol=ncol, nnz=nnz)

    csc_mat.val = val
    csc_mat.ind = ind
    csc_mat.row = row

    csc_mat.__status_ok = True

    return csc_mat