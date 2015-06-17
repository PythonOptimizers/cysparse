"""
Base proxy class to sparse matrices.


"""
from cysparse.sparse.s_mat cimport SparseMatrix

from cysparse.types.cysparse_types cimport CPType

cdef class ProxySparseMatrix:
    """
    Proxy to a :class:`SparseMatrix` object.

    Warning:
        This class is **not** a real matrix.
    """
    cdef:
        public SparseMatrix A

        object nrow
        object ncol
        object dtype
        object itype

        object shape     # for compatibility with numpy, PyKrylov, etc.


