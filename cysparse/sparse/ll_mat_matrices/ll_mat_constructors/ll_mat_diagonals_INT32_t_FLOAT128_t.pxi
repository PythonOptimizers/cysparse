

cdef LLSparseMatrix_INT32_t_FLOAT128_t MakeDiagonalLLSparseMatrix_INT32_t_FLOAT128_t(LLSparseMatrix_INT32_t_FLOAT128_t A, FLOAT128_t element):
    """
    Populate an ``LLSparseMatrix_INT32_t_FLOAT128_t with a number on the main diagonal.

    Note:
        We don't expect the matrix to be square.
    """
    cdef:
        INT32_t i, j, A_nrow, A_ncol

    A_nrow, A_ncol = A.shape

    # NON OPTIMIZED code
    for i from 0 <= i < A_nrow:
        for j from 0 <= j < A_ncol:
            if i == j:
                A.put(i, j, element)

    return A