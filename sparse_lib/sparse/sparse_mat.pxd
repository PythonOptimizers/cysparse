

cdef class SparseMatrix:
    cdef:
        public int nrow  # number of rows
        public int ncol  # number of columns
        public int nnz   # number of values stored

        public bint is_symmetric  # true if symmetric matrix
        public bint store_zeros

        object shape     # for compatibility with numpy, array, etc.

cdef class MutableSparseMatrix(SparseMatrix):
    cdef:
        int size_hint
        int nalloc    # allocated size of mutable 1D arrays


cdef class ImmutableSparseMatrix(SparseMatrix):
    cdef:
        int test2