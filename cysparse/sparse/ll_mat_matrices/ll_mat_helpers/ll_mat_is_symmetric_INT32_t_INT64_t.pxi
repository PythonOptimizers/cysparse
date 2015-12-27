"""
Several helper routines to test for symmetry of an  ``LLSparseMatrix`` matrix.
"""

cdef bint is_symmetric_INT32_t_INT64_t(LLSparseMatrix_INT32_t_INT64_t A):
    """
    Test if an arbitrary :class:``LLSparseMatrix`` matrix is symmetric or not.

    """
    cdef:
        INT32_t i, j, k
        INT64_t v

    for i from 0 <= i < A.nrow:
        k = A.root[i]
        while k != -1:
            j = A.col[k]
            v = A.val[k]

            # test for symmetry
            if v != A[j, i]:
                return False

            k = A.link[k]

    return True