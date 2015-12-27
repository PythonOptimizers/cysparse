"""
Several helper routines to test for symmetry of an  ``CSRSparseMatrix`` matrix.
"""

cdef bint is_symmetric_INT32_t_FLOAT64_t(CSRSparseMatrix_INT32_t_FLOAT64_t A):
    """
    Test if an arbitrary :class:``CSRSparseMatrix`` matrix is symmetric or not.

    """
    cdef:
        INT32_t i, j, k
        FLOAT64_t v

    for i from 0 <= i < A.nrow:
        for k from A.ind[i] <= k < A.ind[i+1]:
            j = A.col[k]
            v = A.val[k]

            # test for symmetry
            if v != A[j, i]:
                return False

    return True