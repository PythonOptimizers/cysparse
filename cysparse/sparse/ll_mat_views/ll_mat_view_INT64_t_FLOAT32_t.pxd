#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
from cysparse.common_types.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView_INT64_t_FLOAT32_t

from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT32_t cimport LLSparseMatrix_INT64_t_FLOAT32_t

from cpython cimport PyObject

cpdef bint PyLLSparseMatrixView_Check(object obj)

cdef class LLSparseMatrixView_INT64_t_FLOAT32_t:
    cdef:
        INT64_t __nrow    # number of rows of the collected view
        INT64_t __ncol    # number of columns of the collected view

        bint __is_empty   # view is empty, probably constructed with bad index objects

        str __full_type_str   # Name of matrix view type
        str __base_type_str   # Type of matrix view

        INT64_t * row_indices  # collected row indices
        INT64_t * col_indices  # collected col indices

        LLSparseMatrix_INT64_t_FLOAT32_t A


    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT64_t i, INT64_t j, FLOAT32_t value)
    cdef int safe_put(self, INT64_t i, INT64_t j, FLOAT32_t value)  except -1
    #cdef assign(self, LLSparseMatrixView view, obj)

    cdef FLOAT32_t at(self, INT64_t i, INT64_t j)
    # EXPLICIT TYPE TESTS

    cdef FLOAT32_t safe_at(self, INT64_t i, INT64_t j) except? 1


cdef LLSparseMatrixView_INT64_t_FLOAT32_t MakeLLSparseMatrixView_INT64_t_FLOAT32_t(LLSparseMatrix_INT64_t_FLOAT32_t A, PyObject* obj1, PyObject* obj2)

cdef LLSparseMatrixView_INT64_t_FLOAT32_t MakeLLSparseMatrixViewFromView_INT64_t_FLOAT32_t(LLSparseMatrixView_INT64_t_FLOAT32_t A, PyObject* obj1, PyObject* obj2)

