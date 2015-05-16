from cysparse.types.cysparse_types cimport *

from cysparse.sparse.sparse_mat_matrices.sparse_mat_INT32_t_FLOAT32_t cimport MutableSparseMatrix_INT32_t_FLOAT32_t
#from cysparse.sparse.ll_mat_view cimport LLSparseMatrixView

cimport numpy as cnp

from cpython cimport PyObject

cdef class LLSparseMatrix_INT32_t_FLOAT32_t(MutableSparseMatrix_INT32_t_FLOAT32_t):
    """
    Linked-List Format matrix.

    Note:
        Despite its name, this matrix doesn't use any linked list.
    """
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    cdef:
        INT32_t  free      # index to first element in free chain
        FLOAT32_t  *val       # pointer to array of values
        INT32_t *col       # pointer to array of indices, see doc
        INT32_t *link      # pointer to array of indices, see doc
        INT32_t *root      # pointer to array of indices, see doc

    cdef _realloc(self, INT32_t nalloc_new)
    cdef _realloc_expand(self)

    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT32_t i, INT32_t j, FLOAT32_t value)
    cdef safe_put(self, INT32_t i, INT32_t j, FLOAT32_t value)
    #cdef assign(self, LLSparseMatrixView view, obj)

    #cdef at(self, INT32_t i, INT32_t j)
    #cdef safe_at(self, INT32_t i, INT32_t j)

    #cdef object _keys(self)
    #cdef object _values(self)
    #cdef object _items(self)

#cdef NewLLSparseMatrix_INT32_t_FLOAT32_t(FLOAT32_t * val, INT32_t *root, INT32_t *link, INT32_t *col, FLOAT32_t * ival)