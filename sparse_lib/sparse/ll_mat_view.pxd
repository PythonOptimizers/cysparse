
# forward declaration
cdef class LLSparseMatrixView

from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

from cpython cimport PyObject


cdef class LLSparseMatrixView:
    cdef:
        public int nrow    # number of rows of the collected view
        public int ncol    # number of columns of the collected view

        public bint is_empty  # view is empty, probably constructed with bad index objects

        int * row_indices  # collected row indices
        int * col_indices  # collected col indices

        LLSparseMatrix A

        bint __status_ok

        public bint is_symmetric
        public bint store_zeros

        object nnz   # number of non zeros elements of the collected view
        bint __counted_nnz  # did we already count the number of nnz?
        int _nnz     # number of non zeros

    cdef int count_nnz(self)
    cdef assert_status_ok(self)
    cdef put(self, int i, int j, double value)
    cdef safe_put(self, int i, int j, double value)

cdef LLSparseMatrixView MakeLLSparseMatrixView(LLSparseMatrix A, PyObject* obj1, PyObject* obj2)

cdef int * create_c_array_indices_from_python_object(int length, PyObject * obj, int * number_of_elements) except NULL

