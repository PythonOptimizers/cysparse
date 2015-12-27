

cdef LLSparseMatrix_INT64_t_COMPLEX64_t MakeLinearFillLLSparseMatrix_INT64_t_COMPLEX64_t(LLSparseMatrix_INT64_t_COMPLEX64_t A, COMPLEX64_t first_element, COMPLEX64_t step, row_wise=True):
    """
    Populate an ``LLSparseMatrix_INT64_t_COMPLEX64_t rows by rows (if ``row_wise`` is ``True``) or columns by columns
    (if ``row_wise`` is ``False``) with a start element ``a`` and successive sums with the addition of a step ``s`` each time:
    ``a``, ``a + s``, ``a + 2s``, etc. until the matrix is completely filled.

    See https://en.wikipedia.org/wiki/Arrowhead_matrix.

    Note:
        This matrix **is** dense. We use it for testing purpose only.
    """
    cdef:
        INT64_t i, j, A_nrow, A_ncol
        COMPLEX64_t sum

    A_nrow, A_ncol = A.shape

    sum = first_element

    # NON OPTIMIZED code
    if A.store_symmetric:
        if row_wise:
            for i from 0 <= i < A_nrow:
                for j from 0 <= j <= i:
                    A.put(i, j, sum)
                    sum += step
        else:  #  column wise
            for i from 0 <= i < A_nrow:
                for j from i <= j < A_ncol:
                    A.put(j, i, sum)
                    sum += step
    else:  # A non symmetric
        if row_wise:
            for i from 0 <= i < A_nrow:
                for j from 0 <= j < A_ncol:
                    A.put(i, j, sum)
                    sum += step

        else:  #  column wise
            for j from 0 <= j < A_ncol:
                for i from 0 <= i < A_nrow:
                    A.put(i, j, sum)
                    sum += step

    return A