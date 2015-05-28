from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_INT32_t cimport MutableSparseMatrix_INT32_t_INT32_t
from cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_INT32_t cimport LLSparseMatrixView_INT32_t_INT32_t

cimport numpy as cnp

from cpython cimport PyObject

cdef class LLSparseMatrix_INT32_t_INT32_t(MutableSparseMatrix_INT32_t_INT32_t):
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
        INT32_t  *val       # pointer to array of values
        INT32_t *col       # pointer to array of indices, see doc
        INT32_t *link      # pointer to array of indices, see doc
        INT32_t *root      # pointer to array of indices, see doc

    cdef _realloc(self, INT32_t nalloc_new)
    cdef _realloc_expand(self)

    ####################################################################################################################
    # CREATE SUB-MATRICES
    ####################################################################################################################
    # TODO: implement method or scrap it
    cdef create_submatrix(self, PyObject* obj1, PyObject* obj2)

    ####################################################################################################################
    # COUNTING ELEMENTS
    ####################################################################################################################
    cdef count_nnz_from_indices(self, INT32_t * row_indices,INT32_t row_indices_length, INT32_t * col_indices, INT32_t col_indices_length)

    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT32_t i, INT32_t j, INT32_t value)
    cdef int safe_put(self, INT32_t i, INT32_t j, INT32_t value) except -1
    cdef assign(self, LLSparseMatrixView_INT32_t_INT32_t view, obj)

    # EXPLICIT TYPE TESTS
    cdef INT32_t at(self, INT32_t i, INT32_t j)

    cdef INT32_t safe_at(self, INT32_t i, INT32_t j) except? 1

    cpdef object keys(self)
    cpdef object values(self)
    cpdef object items(self)
    cpdef find(self)

    ####################################################################################################################
    # Norms
    ####################################################################################################################
    cdef _norm_inf(self)
    cdef _norm_one(self)
    cdef _norm_frob(self)

#cdef INT32_t * create_c_array_indices_from_python_object_INT32_t(INT32_t max_length, PyObject * obj, INT32_t * number_of_elements) except NULL