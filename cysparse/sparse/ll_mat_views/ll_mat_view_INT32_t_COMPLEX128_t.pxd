from cysparse.common_types.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView_INT32_t_COMPLEX128_t

from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX128_t cimport LLSparseMatrix_INT32_t_COMPLEX128_t

from cpython cimport PyObject

cdef class LLSparseMatrixView_INT32_t_COMPLEX128_t:
    cdef:
        INT32_t __nrow    # number of rows of the collected view
        INT32_t __ncol    # number of columns of the collected view

        bint __is_empty   # view is empty, probably constructed with bad index objects

        str __type_name   # Name of matrix view type
        str __type        # Type of matrix view

        INT32_t * row_indices  # collected row indices
        INT32_t * col_indices  # collected col indices

        LLSparseMatrix_INT32_t_COMPLEX128_t A


    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT32_t i, INT32_t j, COMPLEX128_t value)
    cdef int safe_put(self, INT32_t i, INT32_t j, COMPLEX128_t value)  except -1
    #cdef assign(self, LLSparseMatrixView view, obj)

    cdef COMPLEX128_t at(self, INT32_t i, INT32_t j)
    # EXPLICIT TYPE TESTS

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX128_t safe_at(self, INT32_t i, INT32_t j) except *


cdef LLSparseMatrixView_INT32_t_COMPLEX128_t MakeLLSparseMatrixView_INT32_t_COMPLEX128_t(LLSparseMatrix_INT32_t_COMPLEX128_t A, PyObject* obj1, PyObject* obj2)

cdef LLSparseMatrixView_INT32_t_COMPLEX128_t MakeLLSparseMatrixViewFromView_INT32_t_COMPLEX128_t(LLSparseMatrixView_INT32_t_COMPLEX128_t A, PyObject* obj1, PyObject* obj2)

