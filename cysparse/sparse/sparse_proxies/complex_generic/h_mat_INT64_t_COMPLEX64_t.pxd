"""
Syntactic sugar: we want to use the following code: ``A.H * b`` to compute :math:`A^h * b`.

``A.H`` returns a proxy to the original matrix ``A`` that allows us to use the above notation.
"""

from cysparse.types.cysparse_types cimport CPType
from cysparse.sparse.s_mat cimport SparseMatrix

cdef class ConjugateTransposedSparseMatrix_INT64_t_COMPLEX64_t:
    """
    Proxy to the conjugate transposed of a :class:`SparseMatrix`.

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
    cdef:
        object H         # ref to the original matrix

        object T         # ref to A.conj

        public object __A_conj
        object conj      # ref to A.T
