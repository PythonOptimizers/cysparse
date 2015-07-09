"""
Syntactic sugar: we want to use the following code: ``A.T * b`` to compute :math:`A^t * b`.

``A.T`` returns a proxy to the original matrix ``A`` that allows us to use the above notation.
"""

# TODO: verify if we need to generate this file
# For the moment I (Nikolaj) 'm leaving it as it is just in case things change...

from cysparse.types.cysparse_types cimport CPType
from cysparse.sparse.s_mat cimport SparseMatrix

cdef class TransposedSparseMatrix:
    """
    Proxy to the transposed of a :class:`SparseMatrix`.

    Warning:
        This class is **not** a real matrix.
    """
    cdef:

        SparseMatrix A
