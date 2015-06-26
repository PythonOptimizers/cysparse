

cdef LLSparseMatrix_INT32_t_FLOAT64_t MakeLLSparseMatrixArrowHead_INT32_t_FLOAT64_t(LLSparseMatrix_INT32_t_FLOAT64_t A, FLOAT64_t element):
    """
    Populate an ``LLSparseMatrix_INT32_t_FLOAT64_t with a first row, first column and diagonal with a given number.

    See https://en.wikipedia.org/wiki/Arrowhead_matrix.

    Note:
        We don't expect the matrix to be square.
    """

    return A
