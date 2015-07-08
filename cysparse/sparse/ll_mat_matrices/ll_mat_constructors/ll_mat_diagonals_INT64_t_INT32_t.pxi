

cdef LLSparseMatrix_INT64_t_INT32_t MakeDiagonalLLSparseMatrix_INT64_t_INT32_t(LLSparseMatrix_INT64_t_INT32_t A, INT32_t element):
    """
    Populate an ``LLSparseMatrix_INT64_t_INT32_t with a number on the main diagonal.

    Note:
        We don't expect the matrix to be square.
    """
    cdef:
        INT64_t i, j, A_nrow, A_ncol

    A_nrow, A_ncol = A.shape

    # NON OPTIMIZED code
    for i from 0 <= i < A_nrow:
        for j from 0 <= j < A_ncol:
            if i == j:
                A.put(i, j, element)

    return A