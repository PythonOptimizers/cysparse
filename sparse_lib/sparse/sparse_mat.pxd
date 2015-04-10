

cdef class SparseMatrix:
    cdef:
        public int nrow  # number of rows
        public int ncol  # number of columns
        public int nnz   # number of values stored

cdef class MutableSparseMatrix(SparseMatrix):
    cdef:
        int test


cdef class ImmutableSparseMatrix(SparseMatrix):
    cdef:
        int test2