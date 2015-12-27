"""
Main entry point for ``SparseMatrix`` objects.

See also s_mat_matrices/s_mat.* .

"""
from cysparse.common_types.cysparse_types cimport *

#cdef class SparseMatrix

# Use of a "real" factory method, following Robert Bradshaw's suggestion
# https://groups.google.com/forum/#!topic/cython-users/0UHuLqheoq0
cdef unexposed_value

cdef INT32_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT

cdef __set_store_zero_attribute(SparseMatrix A, bint store_zero)
cpdef bint PySparseMatrix_Check(object obj)
cdef bint PyBothSparseMatricesAreOfSameType(object obj1, object obj2)

cdef class SparseMatrix:
    """
    Main base class for sparse matrices.

    Notes:
        This class has been (somewhat arbitrarily) divided in two:

            - this part is minimalistic and generic and doesn't require the types to be known at compile time and
            - s_mat_matrices/s_mat.* (with * in place of 'cpd' or 'cpx') to deal with specifics of types at compile time.

        For instance, we have transferred some definitions into s_mat_matrices/s_mat.* (for instance ``nnz``,
        ``ncol``, ``nrow``). The only rule is to keep everything coherent but basically this
        class and s_mat_matrices/s_mat.* define the same class. Use your judgement.

        This class is also used to break circular dependencies.
    """
    cdef:
        # attributes that have a corresponding Python property start with '__'
        bint __store_symmetric             # True if symmetric matrix
        bint __store_zero                  # True if 0.0 is to be stored explicitly

        bint __is_mutable                  # True if mutable

        str __full_type_str               # Name of matrix type
        str __base_type_str               # Type of matrix
        # the next attribute doesn't have a corresponding Python property by we keep names coherent
        str __index_and_type          # [@index@, @type@]

        CPType cp_type                # Internal types of the matrix



cdef MakeMatrixLikeString(object A, full=?)
