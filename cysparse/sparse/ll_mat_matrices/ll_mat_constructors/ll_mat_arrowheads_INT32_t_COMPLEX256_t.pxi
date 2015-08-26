

cdef LLSparseMatrix_INT32_t_COMPLEX256_t MakeArrowHeadLLSparseMatrix_INT32_t_COMPLEX256_t(LLSparseMatrix_INT32_t_COMPLEX256_t A, COMPLEX256_t element):
    """
    Populate an ``LLSparseMatrix_INT32_t_COMPLEX256_t with a first row, first column and diagonal with a given number.

    See https://en.wikipedia.org/wiki/Arrowhead_matrix.

    Note:
        We don't expect the matrix to be square.
    """
    cdef:
        INT32_t i, j, A_nrow, A_ncol

    A_nrow, A_ncol = A.shape

    # NON OPTIMIZED code
    for i from 0 <= i < A_nrow:
        A.put(i, 0, element)
        for j from 0 <= j < A_ncol:
            A.put(0, j, element)
            if i == j:
                A.put(i, j, element)

    return A
