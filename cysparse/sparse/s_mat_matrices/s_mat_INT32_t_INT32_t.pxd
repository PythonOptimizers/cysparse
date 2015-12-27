from cysparse.common_types.cysparse_types cimport *
from cysparse.sparse.s_mat cimport SparseMatrix

from cysparse.sparse.sparse_proxies.t_mat cimport TransposedSparseMatrix


cdef class SparseMatrix_INT32_t_INT32_t(SparseMatrix):
    cdef:
        INT32_t __nrow  # number of rows
        INT32_t __ncol  # number of columns
        INT32_t __nnz   # number of values stored

        INT32_t __nargin  # size of the input vector in A * b
        INT32_t __nargout # size of the output vector in y = A * b

        # proxy to the transposed matrix
        TransposedSparseMatrix __transposed_proxy_matrix  # transposed matrix proxy
        bint __transposed_proxy_matrix_generated



cdef class MutableSparseMatrix_INT32_t_INT32_t(SparseMatrix_INT32_t_INT32_t):
    cdef:
        INT32_t size_hint # hint to allocate the size of mutable 1D arrays at creation
        INT32_t nalloc    # allocated size of mutable 1D arrays

cdef class ImmutableSparseMatrix_INT32_t_INT32_t(SparseMatrix_INT32_t_INT32_t):
    cdef:
        INT32_t temp