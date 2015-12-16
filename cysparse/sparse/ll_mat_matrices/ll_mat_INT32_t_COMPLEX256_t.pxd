#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from cysparse.common_types.cysparse_types cimport *

from cysparse.sparse.s_mat_matrices.s_mat_INT32_t_COMPLEX256_t cimport MutableSparseMatrix_INT32_t_COMPLEX256_t
# TODO: investigate: how come this could even compile before?
#from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX256_t cimport LLSparseMatrix_INT32_t_COMPLEX256_t
from cysparse.sparse.ll_mat_views.ll_mat_view_INT32_t_COMPLEX256_t cimport LLSparseMatrixView_INT32_t_COMPLEX256_t

cimport numpy as cnp

from cpython cimport PyObject

cdef class LLSparseMatrix_INT32_t_COMPLEX256_t(MutableSparseMatrix_INT32_t_COMPLEX256_t):
    """
    Linked-List Format matrix.

    Note:
        The linked list is made of two C-arrays (``link`` and ``root``).
    """
    ####################################################################################################################
    # Init/Free/Memory
    ####################################################################################################################
    cdef:
        INT32_t  free      # index to first element in free chain
        COMPLEX256_t  *val       # pointer to array of values
        INT32_t *col       # pointer to array of indices, see doc
        INT32_t *link      # pointer to array of indices, see doc
        INT32_t *root      # pointer to array of indices, see doc

    cdef _realloc(self, INT32_t nalloc_new)
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
    cdef INT32_t count_nnz_from_indices(self, INT32_t * row_indices,INT32_t row_indices_length, INT32_t * col_indices,
                                        INT32_t col_indices_length, bint count_only_stored=?)

    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT32_t i, INT32_t j, COMPLEX256_t value)
    cdef safe_put(self, INT32_t i, INT32_t j, COMPLEX256_t value)
    # EXPLICIT TYPE TESTS
    cdef assign(self, LLSparseMatrixView_INT32_t_COMPLEX256_t view, obj)

    # EXPLICIT TYPE TESTS
    cdef COMPLEX256_t at(self, INT32_t i, INT32_t j)

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX256_t safe_at(self, INT32_t i, INT32_t j) except *

    cpdef object keys(self)
    cpdef object values(self)
    cpdef object items(self)

    cdef take_triplet_pointers(self, INT32_t * a_row, INT32_t * a_col, COMPLEX256_t * a_val)

    cpdef take_triplet(self, id1, id2, cnp.ndarray[cnp.npy_complex256, ndim=1] b)
    cpdef put_diagonal(self, INT32_t k, cnp.ndarray[cnp.npy_complex256, ndim=1] b)

    ####################################################################################################################
    # Norms
    ####################################################################################################################
    cdef _norm_inf(self)
    cdef _norm_one(self)
    cdef _norm_frob(self)

#cdef INT32_t * create_c_array_indices_from_python_object_INT32_t(INT32_t max_length, PyObject * obj, INT32_t * number_of_elements) except NULL