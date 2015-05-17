from cysparse.types.cysparse_types cimport *

from cysparse.sparse.sparse_mat_matrices.sparse_mat_INT32_t_FLOAT64_t cimport MutableSparseMatrix_INT32_t_FLOAT64_t
#from cysparse.sparse.ll_mat_view cimport LLSparseMatrixView

cimport numpy as cnp

from cpython cimport PyObject

cdef class LLSparseMatrix_INT32_t_FLOAT64_t(MutableSparseMatrix_INT32_t_FLOAT64_t):
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
        FLOAT64_t  *val       # pointer to array of values
        INT32_t *col       # pointer to array of indices, see doc
        INT32_t *link      # pointer to array of indices, see doc
        INT32_t *root      # pointer to array of indices, see doc

    cdef _realloc(self, INT32_t nalloc_new)
    cdef _realloc_expand(self)

    ####################################################################################################################
    # CREATE SUB-MATRICES
    ####################################################################################################################
    cdef create_submatrix(self, PyObject* obj1, PyObject* obj2)

    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT32_t i, INT32_t j, FLOAT64_t value)
    cdef safe_put(self, INT32_t i, INT32_t j, FLOAT64_t value)
    #cdef assign(self, LLSparseMatrixView view, obj)

    cdef FLOAT64_t at(self, INT32_t i, INT32_t j)
    cdef FLOAT64_t safe_at(self, INT32_t i, INT32_t j)

    cpdef object keys(self)
    cpdef object values(self)
    cpdef object items(self)
    cpdef find(self)



#cdef INT32_t * create_c_array_indices_from_python_object_INT32_t(INT32_t max_length, PyObject * obj, INT32_t * number_of_elements) except NULL