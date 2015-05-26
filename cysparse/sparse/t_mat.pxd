"""
Syntactic sugar: we want to use the following code: ``A.T * b`` to compute :math:`A^t * b`.

``A.T`` returns a proxy to the original matrix ``A`` that allows us to use the above notation.
"""

# forward declaration
cdef class TransposedSparseMatrix

from cysparse.sparse.s_mat cimport SparseMatrix
from cysparse.types.cysparse_types cimport CPType

cdef class TransposedSparseMatrix:
    """
    Proxy to the transposed of a :class:`SparseMatrix`.

    Warning:
        This class is **not** a real matrix.
    """
    cdef:
        public SparseMatrix A

        object nrow
        object ncol
        object dtype
        object itype

        CPType __cp_type # private CPType, only accessible in Cython

        object shape     # for compatibility with numpy, PyKrylov, etc.

        object T         # ref to the original matrix

