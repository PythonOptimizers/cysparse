from cysparse.types.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView_INT64_t_COMPLEX256_t

from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX256_t cimport LLSparseMatrix_INT64_t_COMPLEX256_t

from cpython cimport PyObject


cdef class LLSparseMatrixView_INT64_t_COMPLEX256_t:
    cdef:
        public INT64_t nrow    # number of rows of the collected view
        public INT64_t ncol    # number of columns of the collected view

        public bint is_empty  # view is empty, probably constructed with bad index objects

        public char * __type_name   # Name of matrix view type
        public char * type        # Type of matrix view

        INT64_t * row_indices  # collected row indices
        INT64_t * col_indices  # collected col indices

        LLSparseMatrix_INT64_t_COMPLEX256_t A

        public bint __is_symmetric
        public bint store_zeros

        object nnz   # number of non zeros elements of the collected view
        bint __counted_nnz  # did we already count the number of nnz?
        INT64_t _nnz     # number of non zeros

    ####################################################################################################################
    # SET/GET
    ####################################################################################################################
    cdef put(self, INT64_t i, INT64_t j, COMPLEX256_t value)
    cdef int safe_put(self, INT64_t i, INT64_t j, COMPLEX256_t value)  except -1
    #cdef assign(self, LLSparseMatrixView view, obj)

    cdef COMPLEX256_t at(self, INT64_t i, INT64_t j)
    # EXPLICIT TYPE TESTS

    # this is needed as for the complex type, Cython's compiler crashes...
    cdef COMPLEX256_t safe_at(self, INT64_t i, INT64_t j) except *


cdef LLSparseMatrixView_INT64_t_COMPLEX256_t MakeLLSparseMatrixView_INT64_t_COMPLEX256_t(LLSparseMatrix_INT64_t_COMPLEX256_t A, PyObject* obj1, PyObject* obj2)

cdef LLSparseMatrixView_INT64_t_COMPLEX256_t MakeLLSparseMatrixViewFromView_INT64_t_COMPLEX256_t(LLSparseMatrixView_INT64_t_COMPLEX256_t A, PyObject* obj1, PyObject* obj2)

