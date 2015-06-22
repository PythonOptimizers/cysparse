from cysparse.types.cysparse_types cimport *
from cysparse.sparse.s_mat cimport SparseMatrix

from cysparse.sparse.sparse_proxies.t_mat cimport TransposedSparseMatrix

from cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT32_t_COMPLEX64_t cimport ConjugateTransposedSparseMatrix_INT32_t_COMPLEX64_t
from cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT32_t_COMPLEX64_t cimport ConjugatedSparseMatrix_INT32_t_COMPLEX64_t


cdef class SparseMatrix_INT32_t_COMPLEX64_t(SparseMatrix):
    cdef:
        INT32_t __nrow  # number of rows
        INT32_t __ncol  # number of columns
        INT32_t __nnz   # number of values stored

        INT32_t __nargin  # size of the input vector in A * b
        INT32_t __nargout # size of the output vector in y = A * b

        # proxy to the transposed matrix
        TransposedSparseMatrix __transposed_proxy_matrix  # transposed matrix proxy
        bint __transposed_proxy_matrix_generated


        # proxy to the conjugate transposed matrix
        ConjugateTransposedSparseMatrix_INT32_t_COMPLEX64_t __conjugate_transposed_proxy_matrix
        bint __conjugate_transposed_proxy_matrix_generated

        # proxy to the conjugated matrix
        ConjugatedSparseMatrix_INT32_t_COMPLEX64_t __conjugated_proxy_matrix
        bint __conjugated_proxy_matrix_generated


cdef class MutableSparseMatrix_INT32_t_COMPLEX64_t(SparseMatrix_INT32_t_COMPLEX64_t):
    cdef:
        INT32_t size_hint # hint to allocate the size of mutable 1D arrays at creation
        INT32_t nalloc    # allocated size of mutable 1D arrays

cdef class ImmutableSparseMatrix_INT32_t_COMPLEX64_t(SparseMatrix_INT32_t_COMPLEX64_t):
    cdef:
        INT32_t test2