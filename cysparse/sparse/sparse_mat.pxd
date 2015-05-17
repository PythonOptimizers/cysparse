from cysparse.types.cysparse_types cimport *

# Use of a "real" factory method, following Robert Bradshaw's suestion
# https://groups.google.com/forum/#!topic/cython-users/0UHuLqheoq0
cdef unexposed_value

cdef INT32_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT

cdef class SparseMatrix:
    cdef:
       
        public bint is_symmetric  # True if symmetric matrix
        public bint store_zeros   # True if 0.0 is to be stored explicitly

        bint is_mutable           # True if mutable

        public char * type_name   # Name of matrix type
        CPType cp_type            # Internal types of the matrix

        object shape     # for compatibility with numpy, array, etc.

        object T         # for the transposed matrix

