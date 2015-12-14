"""
Syntactic sugar: we want to use the following code: ``A.conj * b`` to compute :math:`\textrm{conj}(A) * b`.

``A.conj`` returns a proxy to the original matrix ``A`` that allows us to use the above notation.
"""

from cysparse.cysparse_types.cysparse_types cimport CPType
from cysparse.sparse.s_mat cimport SparseMatrix

cdef class ConjugatedSparseMatrix_INT64_t_COMPLEX256_t:
    """
    Proxy to the conjugated of a :class:`SparseMatrix`.

    Warning:
        This class is **not** a real matrix.
    """
    cdef:

        SparseMatrix A