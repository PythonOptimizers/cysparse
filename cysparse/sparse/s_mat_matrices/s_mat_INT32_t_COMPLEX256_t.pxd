from cysparse.types.cysparse_types cimport *
from cysparse.sparse.s_mat cimport SparseMatrix


from cysparse.sparse.sparse_proxies.complex_generic.h_mat_INT32_t_COMPLEX256_t cimport ConjugateTransposedSparseMatrix_INT32_t_COMPLEX256_t
from cysparse.sparse.sparse_proxies.complex_generic.conj_mat_INT32_t_COMPLEX256_t cimport ConjugatedSparseMatrix_INT32_t_COMPLEX256_t


cdef class SparseMatrix_INT32_t_COMPLEX256_t(SparseMatrix):
    cdef:
        public INT32_t nrow  # number of rows
        public INT32_t ncol  # number of columns
        public INT32_t nnz   # number of values stored


        object H         # proxy to the conjugate transposed matrix

        ConjugateTransposedSparseMatrix_INT32_t_COMPLEX256_t __conjugate_transposed_proxy_matrix
        bint __conjugate_transposed_proxy_matrix_generated

        object conj         # proxy to the conjugated matrix

        ConjugatedSparseMatrix_INT32_t_COMPLEX256_t __conjugated_proxy_matrix
        bint __conjugated_proxy_matrix_generated




cdef class MutableSparseMatrix_INT32_t_COMPLEX256_t(SparseMatrix_INT32_t_COMPLEX256_t):
    cdef:
        INT32_t size_hint # hint to allocate the size of mutable 1D arrays at creation
        INT32_t nalloc    # allocated size of mutable 1D arrays

cdef class ImmutableSparseMatrix_INT32_t_COMPLEX256_t(SparseMatrix_INT32_t_COMPLEX256_t):
    cdef:
        INT32_t test2