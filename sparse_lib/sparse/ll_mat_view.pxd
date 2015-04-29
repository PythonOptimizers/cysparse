from sparse_lib.cysparse_types cimport *

# forward declaration
cdef class LLSparseMatrixView

from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

from cpython cimport PyObject


cdef class LLSparseMatrixView:
    cdef:
        public INT_t nrow    # number of rows of the collected view
        public INT_t ncol    # number of columns of the collected view

        public bint is_empty  # view is empty, probably constructed with bad index objects

        INT_t * row_indices  # collected row indices
        INT_t * col_indices  # collected col indices

        LLSparseMatrix A

        public bint is_symmetric
        public bint store_zeros

        object nnz   # number of non zeros elements of the collected view
        bint __counted_nnz  # did we already count the number of nnz?
        INT_t _nnz     # number of non zeros

    cdef INT_t _count_nnz(self)
    cdef at(self, INT_t i, INT_t j)
    cdef safe_at(self, INT_t i, INT_t j)
    cdef put(self, INT_t i, INT_t j, double value)
    cdef safe_put(self, INT_t i, INT_t j, double value)

cdef LLSparseMatrixView MakeLLSparseMatrixView(LLSparseMatrix A, PyObject* obj1, PyObject* obj2)

cdef LLSparseMatrixView MakeLLSparseMatrixViewFromView(LLSparseMatrixView A, PyObject* obj1, PyObject* obj2)


cdef INT_t * create_c_array_indices_from_python_object(INT_t length, PyObject * obj, INT_t * number_of_elements) except NULL

