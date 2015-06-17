"""
Syntactic sugar: we want to use the following code: ``A.T * b`` to compute :math:`A^t * b`.

``A.T`` returns a proxy to the original matrix ``A`` that allows us to use the above notation.
"""
cdef class TransposedSparseMatrix

from cysparse.types.cysparse_types cimport CPType
from cysparse.sparse.s_mat cimport SparseMatrix

cdef class TransposedSparseMatrix:
    """
    Proxy to the transposed of a :class:`SparseMatrix`.

    Warning:
        This class is **not** a real matrix.
    """
    ####################################################################################################################
    # COMMON CODE FROM proxy_common_pxd.txt: see #113
    ####################################################################################################################
    cdef:

        public SparseMatrix A

        object nrow
        object ncol
        object dtype
        object itype

        object shape     # for compatibility with numpy, PyKrylov, etc.

    cdef:
        object T         # ref to the original matrix

