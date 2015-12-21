from cysparse.common_types.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView_INT32_t_COMPLEX256_t

from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX256_t cimport LLSparseMatrix_INT32_t_COMPLEX256_t

from cpython cimport PyObject

cpdef bint PyLLSparseMatrixView_Check(object obj)

cdef class LLSparseMatrixView_INT32_t_COMPLEX256_t:
    cdef:
        INT32_t __nrow    # number of rows of the collected view
        INT32_t __ncol    # number of columns of the collected view

        bint __is_empty   # view is empty, probably constructed with bad index objects

        str __full_type_str   # Name of matrix view type
        str __base_type_str   # Type of matrix view

        INT32_t * row_indices  # collected row indices
        INT32_t * col_indices  # collected col indices

        LLSparseMatrix_INT32_t_COMPLEX256_t A


    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT32_t i, INT32_t j, COMPLEX256_t value)
    cdef int safe_put(self, INT32_t i, INT32_t j, COMPLEX256_t value)  except -1
    #cdef assign(self, LLSparseMatrixView view, obj)

    cdef COMPLEX256_t at(self, INT32_t i, INT32_t j)
    # EXPLICIT TYPE TESTS

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX256_t safe_at(self, INT32_t i, INT32_t j) except *


cdef LLSparseMatrixView_INT32_t_COMPLEX256_t MakeLLSparseMatrixView_INT32_t_COMPLEX256_t(LLSparseMatrix_INT32_t_COMPLEX256_t A, PyObject* obj1, PyObject* obj2)

cdef LLSparseMatrixView_INT32_t_COMPLEX256_t MakeLLSparseMatrixViewFromView_INT32_t_COMPLEX256_t(LLSparseMatrixView_INT32_t_COMPLEX256_t A, PyObject* obj1, PyObject* obj2)

