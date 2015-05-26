"""
Syntactic sugar: we want to use the following code: ``A.T * b`` to compute :math:`A^t * b`.

``A.T`` returns a proxy to the original matrix ``A`` that allows us to use the above notation.
"""

# forward declaration
cdef class TransposedSparseMatrix

from cysparse.sparse.sparse_mat cimport SparseMatrix

cdef class TransposedSparseMatrix:
    """
    Proxy to the transposed of a :class:`SparseMatrix`.

    Warning:
        This class is **not** a real matrix.
    """
    cdef:
        public SparseMatrix A

        object T         # ref to the original matrix

