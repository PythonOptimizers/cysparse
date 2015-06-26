

cdef LLSparseMatrix_INT64_t_INT32_t MakeLLSparseMatrixArrowHead_INT64_t_INT32_t(LLSparseMatrix_INT64_t_INT32_t A, INT32_t element):
    """
    Populate an ``LLSparseMatrix_INT64_t_INT32_t with a first row, first column and diagonal with a given number.

    See https://en.wikipedia.org/wiki/Arrowhead_matrix.

    Note:
        We don't expect the matrix to be square.
    """

    return A
