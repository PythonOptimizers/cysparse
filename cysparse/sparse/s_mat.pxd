"""
Main entry point for ``SparseMatrix`` objects.

See also s_mat_matrices/s_mat.* .

"""
from cysparse.cysparse_types.cysparse_types cimport *

#cdef class SparseMatrix

# Use of a "real" factory method, following Robert Bradshaw's suggestion
# https://groups.google.com/forum/#!topic/cython-users/0UHuLqheoq0
cdef unexposed_value

cdef INT32_t MUTABLE_SPARSE_MAT_DEFAULT_SIZE_HINT

cdef __set_use_zero_storage_attribute(SparseMatrix A, bint use_zero_storage)
cpdef bint PySparseMatrix_Check(object obj)
cpdef bint PyLLSparseMatrixView_Check(object obj)

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
        # attributes that have a corresponding Python property start with '__'
        bint __use_symmetric_storage           # True if symmetric matrix
        bint __use_zero_storage            # True if 0.0 is to be stored explicitly

        bint __is_mutable             # True if mutable

        str __type_name               # Name of matrix type
        str __type                    # Type of matrix
        # the next attribute doesn't have a corresponding Python property by we keep names coherent
        str __index_and_type          # [@index@, @type@]

        CPType cp_type                # Internal types of the matrix


cdef MakeMatrixString(object A, full=?)
