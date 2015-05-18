from cysparse.types.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView_INT32_t_FLOAT32_t

from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT32_t cimport LLSparseMatrix_INT32_t_FLOAT32_t

from cpython cimport PyObject


cdef class LLSparseMatrixView_INT32_t_FLOAT32_t:
    cdef:
        public INT32_t nrow    # number of rows of the collected view
        public INT32_t ncol    # number of columns of the collected view

        public bint is_empty  # view is empty, probably constructed with bad index objects

        public char * type_name   # Name of matrix view type
        public char * type        # Type of matrix view

        INT32_t * row_indices  # collected row indices
        INT32_t * col_indices  # collected col indices

        LLSparseMatrix_INT32_t_FLOAT32_t A

        public bint is_symmetric
        public bint store_zeros

        object nnz   # number of non zeros elements of the collected view
        bint __counted_nnz  # did we already count the number of nnz?
        INT32_t _nnz     # number of non zeros

    #cdef INT32_t _count_nnz(self)
    #cdef at(self, INT32_t i, INT32_t j)
    #cdef safe_at(self, INT32_t i, INT32_t j)
    #cdef put(self, INT32_t i, INT32_t j, FLOAT32_t value)
    #cdef safe_put(self, INT32_t i, INT32_t j, FLOAT32_t value)

cdef LLSparseMatrixView_INT32_t_FLOAT32_t MakeLLSparseMatrixView_INT32_t_FLOAT32_t(LLSparseMatrix_INT32_t_FLOAT32_t A, PyObject* obj1, PyObject* obj2)

#cdef LLSparseMatrixView_INT32_t_FLOAT32_t MakeLLSparseMatrixViewFromView_INT32_t_FLOAT32_t(LLSparseMatrixView_INT32_t_FLOAT32_t A, PyObject* obj1, PyObject* obj2)

