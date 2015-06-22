"""
Main entry point for ``SparseMatrix`` objects.

See also s_mat_matrices/s_mat.* .

"""
from cysparse.types.cysparse_types cimport *

cdef class SparseMatrix

# Use of a "real" factory method, following Robert Bradshaw's suggestion
# https://groups.google.com/forum/#!topic/cython-users/0UHuLqheoq0
cdef unexposed_value

cdef INT32_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT

cdef __set_store_zeros_attribute(SparseMatrix A, bint store_zeros)
cpdef bint PySparseMatrix_Check(object obj)

cdef class SparseMatrix:
    """
    Main base class for sparse matrices.

    Notes:
        This class has been (somewhat arbitrarily) divided in two:

            - this part is minimalistic and generic and doesn't require the types to be known at compile time and
            - s_mat_matrices/s_mat.* (with * in place of 'cpd' or 'cpx') to deal with specifics of types at compile time.

        For instance, we have transferred some definitions into s_mat_matrices/s_mat.* (for instance ``nnz``,
        ``ncol``, ``nrow``) because we also define the mutable/immutable versions of sparse matrices. The only rule
        is to keep everything coherent but basically this class and s_mat_matrices/s_mat.* define the same class. Use your judgement.

        This class is also used to break circular dependencies.
    """
    cdef:
        bint __is_symmetric       # True if symmetric matrix
        bint __store_zeros        # True if 0.0 is to be stored explicitly

        bint __is_mutable         # True if mutable

        char * __type_name        # Name of matrix type
        char * __type             # Type of matrix
        CPType cp_type            # Internal types of the matrix



