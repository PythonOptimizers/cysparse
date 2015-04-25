cdef LLSparseMatrix transposed_ll_mat(LLSparseMatrix A):
    """
    Compute transposed matrix.

    Args:
        A: A :class:`LLSparseMatrix` :math:`A`.

    Note:
        The transposed matrix uses the same amount of internal memory as the

    Returns:
        The corresponding transposed :math:`A^t` :class:`LLSparseMatrix`.
    """
    # TODO: optimize to pure Cython code
    if A.is_symmetric:
        raise NotImplemented("Transposed is not implemented yet for symmetric matrices")

    cdef:
        INT_t A_nrow = A.nrow
        INT_t A_ncol = A.ncol

        INT_t At_nrow = A.ncol
        INT_t At_ncol = A.nrow

        INT_t At_nalloc = A.nalloc

        INT_t i, k
        FLOAT_t val

    cdef LLSparseMatrix transposed_A = LLSparseMatrix(nrow =At_nrow, ncol=At_ncol, size_hint=At_nalloc)

    for i from 0 <= i < A_nrow:
        k = A.root[i]

        while k != -1:
            val = A.val[k]
            j = A.col[k]
            k = A.link[k]

            transposed_A[j, i] = val


    return transposed_A
