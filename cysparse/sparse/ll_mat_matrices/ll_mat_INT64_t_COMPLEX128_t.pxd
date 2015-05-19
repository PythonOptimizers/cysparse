from cysparse.types.cysparse_types cimport *

from cysparse.sparse.sparse_mat_matrices.sparse_mat_INT64_t_COMPLEX128_t cimport MutableSparseMatrix_INT64_t_COMPLEX128_t
from cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_COMPLEX128_t cimport LLSparseMatrixView_INT64_t_COMPLEX128_t

cimport numpy as cnp

from cpython cimport PyObject

cdef class LLSparseMatrix_INT64_t_COMPLEX128_t(MutableSparseMatrix_INT64_t_COMPLEX128_t):
    """
    Linked-List Format matrix.

    Note:
        Despite its name, this matrix doesn't use any linked list.
    """
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    cdef:
        INT64_t  free      # index to first element in free chain
        COMPLEX128_t  *val       # pointer to array of values
        INT64_t *col       # pointer to array of indices, see doc
        INT64_t *link      # pointer to array of indices, see doc
        INT64_t *root      # pointer to array of indices, see doc

    cdef _realloc(self, INT64_t nalloc_new)
    cdef _realloc_expand(self)

    ####################################################################################################################
    # CREATE SUB-MATRICES
    ####################################################################################################################
    # TODO: implement method or scrap it
    cdef create_submatrix(self, PyObject* obj1, PyObject* obj2)

    ####################################################################################################################
    # COUNTING ELEMENTS
    ####################################################################################################################
    cdef count_nnz_from_indices(self, INT64_t * row_indices,INT64_t row_indices_length, INT64_t * col_indices, INT64_t col_indices_length)

    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT64_t i, INT64_t j, COMPLEX128_t value)
    cdef safe_put(self, INT64_t i, INT64_t j, COMPLEX128_t value)
    cdef assign(self, LLSparseMatrixView_INT64_t_COMPLEX128_t view, obj)

    cdef COMPLEX128_t at(self, INT64_t i, INT64_t j)
    cdef COMPLEX128_t safe_at(self, INT64_t i, INT64_t j)

    cpdef object keys(self)
    cpdef object values(self)
    cpdef object items(self)
    cpdef find(self)



#cdef INT64_t * create_c_array_indices_from_python_object_INT64_t(INT64_t max_length, PyObject * obj, INT64_t * number_of_elements) except NULL