"""
Several helper routines to test for symmetry of an  ``CSRSparseMatrix`` matrix.
"""

cdef bint is_symmetric_INT64_t_INT64_t(CSRSparseMatrix_INT64_t_INT64_t A):
    """
    Test if an arbitrary :class:``CSRSparseMatrix`` matrix is symmetric or not.

    """
    cdef:
        INT64_t i, j, k
        INT64_t v

    for i from 0 <= i < A.nrow:
        for k from A.ind[i] <= k < A.ind[i+1]:
            j = A.col[k]
            v = A.val[k]

            # test for symmetry
            if v != A[j, i]:
                return False

    return True