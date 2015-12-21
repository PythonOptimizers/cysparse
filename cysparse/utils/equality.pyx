from cysparse.sparse.ll_mat cimport LLSparseMatrix

cdef SPARSE_LIB_PRECISION = 0.0001

cpdef bint values_are_equal(double x, double y):
    # TODO: get rid of this function
    return abs(x - y) < SPARSE_LIB_PRECISION


cpdef bint ll_mats_are_equals(LLSparseMatrix A, LLSparseMatrix B):
    """
    Test if two :class:`LLSparseMatrix` are equal or not.

    By equal, we mean representing the same matrix. We don't care if one matrix is symmetric and the other not,
    or if they may contain 0 or not.

    Args:
        A: A :class:`LLSparseMatrix` to compare.
        B: Another :class:`LLSparseMatrix` to compare.

    Returns:
        ``True`` if both matrix are similar, ``False`` otherwise.
    """
    # test for dimensions
    if A.nrow != B.nrow or A.ncol != B.ncol:
        return 0

    cdef nrow = A.nrow
    cdef ncol = A.ncol
    cdef int i, j

    # test elements
    if not A.store_zero and not B.store_zero and not A.__store_symmetric and not B.__store_symmetric:
        if A.nnz != B.nnz:
            return 0

    for i from 0 <= i < nrow:
        for j from 0 <= j < ncol:
            if not values_are_equal(A.at(i, j), B.at(i, j)):
                return 0

    return 1


