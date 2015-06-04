"""
Condensed Sparse Row (CSR) Format Matrices.


"""

from __future__ import print_function

from cysparse.sparse.s_mat cimport unexposed_value

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport PyObject

cdef class CSRSparseMatrix_INT64_t_FLOAT64_t(ImmutableSparseMatrix_INT64_t_FLOAT64_t):
    """
    Compressed Sparse Row Format matrix.

    Note:
        This matrix can **not** be modified.

    """
    ####################################################################################################################
    # Init/Free
    ####################################################################################################################
    def __cinit__(self, **kwargs):

        self.type_name = "CSRSparseMatrix"

    def __dealloc__(self):
        PyMem_Free(self.val)
        PyMem_Free(self.col)
        PyMem_Free(self.ind)


########################################################################################################################
# Factory methods
########################################################################################################################
cdef MakeCSRSparseMatrix_INT64_t_FLOAT64_t(INT64_t nrow, INT64_t ncol, INT64_t nnz, INT64_t * ind, INT64_t * col, FLOAT64_t * val):
    """
    Construct a CSRSparseMatrix object.

    Args:
        nrow (INT64_t): Number of rows.
        ncol (INT64_t): Number of columns.
        nnz (INT64_t): Number of non-zeros.
        ind (INT64_t *): C-array with column indices pointers.
        col  (INT64_t *): C-array with column indices.
        val  (FLOAT64_t *): C-array with values.
    """


    csr_mat = CSRSparseMatrix_INT64_t_FLOAT64_t(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz)

    csr_mat.val = val
    csr_mat.ind = ind
    csr_mat.col = col

    return csr_mat
