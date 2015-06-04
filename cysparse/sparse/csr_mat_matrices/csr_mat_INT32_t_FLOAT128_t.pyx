"""
Condensed Sparse Row (CSR) Format Matrices.


"""

from __future__ import print_function

from cysparse.sparse.s_mat cimport unexposed_value

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport PyObject

cdef class CSRSparseMatrix_INT32_t_FLOAT128_t(ImmutableSparseMatrix_INT32_t_FLOAT128_t):
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
cdef MakeCSRSparseMatrix_INT32_t_FLOAT128_t(INT32_t nrow, INT32_t ncol, INT32_t nnz, INT32_t * ind, INT32_t * col, FLOAT128_t * val):
    """
    Construct a CSRSparseMatrix object.

    Args:
        nrow (INT32_t): Number of rows.
        ncol (INT32_t): Number of columns.
        nnz (INT32_t): Number of non-zeros.
        ind (INT32_t *): C-array with column indices pointers.
        col  (INT32_t *): C-array with column indices.
        val  (FLOAT128_t *): C-array with values.
    """


    csr_mat = CSRSparseMatrix_INT32_t_FLOAT128_t(control_object=unexposed_value, nrow=nrow, ncol=ncol, nnz=nnz)

    csr_mat.val = val
    csr_mat.ind = ind
    csr_mat.col = col

    return csr_mat
