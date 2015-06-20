from cysparse.types.cysparse_types cimport *
from cysparse.sparse.s_mat cimport SparseMatrix

from cysparse.sparse.sparse_proxies.t_mat cimport TransposedSparseMatrix

from cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT64_t_COMPLEX256_t cimport ConjugateTransposedSparseMatrix_INT64_t_COMPLEX256_t
from cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT64_t_COMPLEX256_t cimport ConjugatedSparseMatrix_INT64_t_COMPLEX256_t


cdef class SparseMatrix_INT64_t_COMPLEX256_t(SparseMatrix):
    cdef:
        public INT64_t nrow  # number of rows
        public INT64_t ncol  # number of columns
        public INT64_t nnz   # number of values stored

        # proxy to the transposed matrix
        TransposedSparseMatrix __transposed_proxy_matrix  # transposed matrix proxy
        bint __transposed_proxy_matrix_generated


        # proxy to the conjugate transposed matrix
        ConjugateTransposedSparseMatrix_INT64_t_COMPLEX256_t __conjugate_transposed_proxy_matrix
        bint __conjugate_transposed_proxy_matrix_generated

        # proxy to the conjugated matrix
        ConjugatedSparseMatrix_INT64_t_COMPLEX256_t __conjugated_proxy_matrix
        bint __conjugated_proxy_matrix_generated


cdef class MutableSparseMatrix_INT64_t_COMPLEX256_t(SparseMatrix_INT64_t_COMPLEX256_t):
    cdef:
        INT64_t size_hint # hint to allocate the size of mutable 1D arrays at creation
        INT64_t nalloc    # allocated size of mutable 1D arrays

cdef class ImmutableSparseMatrix_INT64_t_COMPLEX256_t(SparseMatrix_INT64_t_COMPLEX256_t):
    cdef:
        INT64_t test2