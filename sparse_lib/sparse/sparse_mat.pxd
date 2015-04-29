from sparse_lib.cysparse_types cimport *

# Use of a "real" factory method, following Robert Bradshaw's suggestion
# https://groups.google.com/forum/#!topic/cython-users/0UHuLqheoq0
cdef unexposed_value

cdef class SparseMatrix:
    cdef:
        public INT_t nrow  # number of rows
        public INT_t ncol  # number of columns
        public INT_t nnz   # number of values stored

        public bint is_symmetric  # True if symmetric matrix
        public bint store_zeros   # True if 0.0 is to be stored explicitly
        public bint is_complex    # True if values are complex

        public char * type_name

        object shape     # for compatibility with numpy, array, etc.

        object T         # for the transposed matrix

cdef class MutableSparseMatrix(SparseMatrix):
    cdef:
        INT_t size_hint
        INT_t nalloc    # allocated size of mutable 1D arrays


cdef class ImmutableSparseMatrix(SparseMatrix):
    cdef:
        INT_t test2