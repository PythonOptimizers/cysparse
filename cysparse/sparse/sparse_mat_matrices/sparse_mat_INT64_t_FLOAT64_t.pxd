from cysparse.types.cysparse_types cimport *
from cysparse.sparse.sparse_mat cimport SparseMatrix


cdef class SparseMatrix_INT64_t_FLOAT64_t(SparseMatrix):
    cdef:
        public INT64_t nrow  # number of rows
        public INT64_t ncol  # number of columns
        public INT64_t nnz   # number of values stored


cdef class MutableSparseMatrix_INT64_t_FLOAT64_t(SparseMatrix_INT64_t_FLOAT64_t):
    cdef:
        INT64_t size_hint # hint to allocate the size of mutable 1D arrays at creation
        INT64_t nalloc    # allocated size of mutable 1D arrays

cdef class ImmutableSparseMatrix_INT64_t_FLOAT64_t(SparseMatrix_INT64_t_FLOAT64_t):
    cdef:
        INT64_t test2