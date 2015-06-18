from cysparse.types.cysparse_types cimport *

cdef class SparseMatrix

from cysparse.sparse.sparse_proxies.t_mat cimport TransposedSparseMatrix

# Use of a "real" factory method, following Robert Bradshaw's suggestion
# https://groups.google.com/forum/#!topic/cython-users/0UHuLqheoq0
cdef unexposed_value

cdef INT32_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT

cpdef bint PySparseMatrix_Check(object obj)

cdef class SparseMatrix:
    cdef:
       
        public bint is_symmetric  # True if symmetric matrix
        public bint store_zeros   # True if 0.0 is to be stored explicitly

        bint is_mutable           # True if mutable

        public char * type_name   # Name of matrix type
        public char * type        # Type of matrix
        CPType cp_type            # Internal types of the matrix

        # Transposed Proxy (conjugate transposed is defined in specialised SparseMatrix_@index@_@type@ but **only** for complex element types)
        object T         # proxy to the transposed matrix

        TransposedSparseMatrix __transposed_proxy_matrix  # transposed matrix proxy
        bint __transposed_proxy_matrix_generated

        object shape     # for compatibility with numpy, PyKrylov, etc.

        object dtype
        object itype


