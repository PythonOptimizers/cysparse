from cysparse.types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT64_t_COMPLEX128_t cimport MutableSparseMatrix_INT64_t_COMPLEX128_t
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX128_t cimport LLSparseMatrix_INT64_t_COMPLEX128_t
from cysparse.sparse.ll_mat_views.ll_mat_view_INT64_t_COMPLEX128_t cimport LLSparseMatrixView_INT64_t_COMPLEX128_t

cimport numpy as cnp

from cpython cimport PyObject

cdef class LLSparseMatrix_INT64_t_COMPLEX128_t(MutableSparseMatrix_INT64_t_COMPLEX128_t):
    """
    Linked-List Format matrix.

    Note:
        The linked list is made of two C-arrays (``link`` and ``root``).
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
    # SORTING
    ####################################################################################################################
    cdef bint is_sorted(self)

    ####################################################################################################################
    # CREATE SUB-MATRICES
    ####################################################################################################################
    # TODO: implement method or scrap it
    cdef create_submatrix(self, PyObject* obj1, PyObject* obj2)

    ####################################################################################################################
    # COUNTING ELEMENTS
    ####################################################################################################################
    cdef INT64_t count_nnz_from_indices(self, INT64_t * row_indices,INT64_t row_indices_length, INT64_t * col_indices, INT64_t col_indices_length)

    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT64_t i, INT64_t j, COMPLEX128_t value)
    cdef int safe_put(self, INT64_t i, INT64_t j, COMPLEX128_t value) except -1
    cdef assign(self, LLSparseMatrixView_INT64_t_COMPLEX128_t view, obj)

    # EXPLICIT TYPE TESTS
    cdef COMPLEX128_t at(self, INT64_t i, INT64_t j)

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX128_t safe_at(self, INT64_t i, INT64_t j) except *

    cpdef object keys(self)
    cpdef object values(self)
    cpdef object items(self)

    cpdef take_triplet(self, id1, id2, cnp.ndarray[cnp.npy_complex128, ndim=1] b)
    cpdef put_diagonal(self, INT64_t k, cnp.ndarray[cnp.npy_complex128, ndim=1] b)

    ####################################################################################################################
    # Norms
    ####################################################################################################################
    cdef _norm_inf(self)
    cdef _norm_one(self)
    cdef _norm_frob(self)

#cdef INT64_t * create_c_array_indices_from_python_object_INT64_t(INT64_t max_length, PyObject * obj, INT64_t * number_of_elements) except NULL