from cysparse.types.cysparse_types cimport *
from cysparse.sparse.s_mat cimport SparseMatrix


cdef class SparseMatrix_INT32_t_FLOAT64_t(SparseMatrix):
    cdef:
        public INT32_t nrow  # number of rows
        public INT32_t ncol  # number of columns
        public INT32_t nnz   # number of values stored


cdef class MutableSparseMatrix_INT32_t_FLOAT64_t(SparseMatrix_INT32_t_FLOAT64_t):
    cdef:
        INT32_t size_hint # hint to allocate the size of mutable 1D arrays at creation
        INT32_t nalloc    # allocated size of mutable 1D arrays

cdef class ImmutableSparseMatrix_INT32_t_FLOAT64_t(SparseMatrix_INT32_t_FLOAT64_t):
    cdef:
        INT32_t test2