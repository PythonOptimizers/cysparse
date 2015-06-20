"""
Syntactic sugar: we want to use the following code: ``A.conj * b`` to compute :math:`\textrm{conj}(A) * b`.

``A.conj`` returns a proxy to the original matrix ``A`` that allows us to use the above notation.
"""

from cysparse.types.cysparse_types cimport CPType
from cysparse.sparse.s_mat cimport SparseMatrix

cdef class ConjugatedSparseMatrix_INT64_t_COMPLEX64_t:
    """
    Proxy to the conjugated of a :class:`SparseMatrix`.

    Warning:
        This class is **not** a real matrix.
    """
    ####################################################################################################################
    # Common code from p_mat.pxd See #113: I could not solve the circular dependencies...
    ####################################################################################################################
    cdef:

        public SparseMatrix A

        object nrow
        object ncol
        object dtype
        object itype

        object shape     # for compatibility with numpy, PyKrylov, etc.

    ####################################################################################################################
    # End of Common code
    ####################################################################################################################
    #cdef:

    #    object conj      # ref to the original matrix

    #    object H         # ref to A.T
    #    object T         # ref to A.H
