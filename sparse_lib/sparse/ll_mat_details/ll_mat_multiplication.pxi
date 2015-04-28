
########################################################################################################################
# Multiplication functions
########################################################################################################################
cdef LLSparseMatrix multiply_two_ll_mat(LLSparseMatrix A, LLSparseMatrix B):
    """
    Multiply two :class:`LLSparseMatrix` ``A`` and ``B``.

    Args:
        A: An :class:``LLSparseMatrix`` ``A``.
        B: An :class:``LLSparseMatrix`` ``B``.

    Returns:
        A **new** :class:``LLSparseMatrix`` ``C = A * B``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplemented``: When matrix ``A`` or ``B`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    # TODO: LLSparseMatrix * A, LLSparseMatrix * B ...
    # test dimensions
    cdef INT_t A_nrow = A.nrow
    cdef INT_t A_ncol = A.ncol

    cdef INT_t B_nrow = B.nrow
    cdef INT_t B_ncol = B.ncol

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT_t C_nrow = A_nrow
    cdef INT_t C_ncol = B_ncol

    cdef bint store_zeros = A.store_zeros and B.store_zeros
    cdef INT_t size_hint = A.size_hint

    C = LLSparseMatrix(nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zeros=store_zeros)


    # CASES
    if not A.is_symmetric and not B.is_symmetric:
        pass
    else:
        raise NotImplemented("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    cdef:
        FLOAT_t valA
        INT_t iA, jA, kA, kB

    for iA from 0 <= iA < A_nrow:
        kA = A.root[iA]

        while kA != -1:
            valA = A.val[kA]
            jA = A.col[kA]
            kA = A.link[kA]

            # add jA-th row of B to iA-th row of C
            kB = B.root[jA]
            while kB != -1:
                update_ll_mat_item_add(C, iA, B.col[kB], valA*B.val[kB])
                kB = B.link[kB]
    return C

cdef multiply_ll_mat_with_numpy_ndarray(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=2] B):
    raise NotImplemented("Multiplication with numpy ndarray of dim 2 not implemented yet")

cdef cnp.ndarray[cnp.double_t, ndim=1] multiply_ll_mat_with_numpy_vector(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=1, mode="c"] b):
    """
    Multiply a :class:`LLSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`LLSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        ``c = A * b``: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    """
    # TODO: take strides into account!
    # test if numpy array is c-contiguous

    cdef INT_t A_nrow = A.nrow
    cdef INT_t A_ncol = A.ncol

    #temp = cnp.NPY_DOUBLE

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    #cdef FLOAT_t * b_data = <FLOAT_t *> b.data
    cdef FLOAT_t * b_data = <FLOAT_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    cdef cnp.ndarray[cnp.double_t, ndim=1] c = np.empty(A_nrow, dtype=np.float64)
    #cdef FLOAT_t * c_data = <FLOAT_t *> c.data
    cdef FLOAT_t * c_data = <FLOAT_t *> cnp.PyArray_DATA(c)

    cdef:
        INT_t i, j
        INT_t k

        FLOAT_t val
        FLOAT_t val_c

    for i from 0 <= i < A_nrow:
        k = A.root[i]

        val_c = 0.0

        while k != -1:
            val = A.val[k]
            j = A.col[k]
            k = A.link[k]

            val_c += val * b_data[j]

        c_data[i] = val_c


    return c


cdef cnp.ndarray[cnp.double_t, ndim=1, mode='c'] multiply_ll_mat_with_numpy_vector2(LLSparseMatrix A, cnp.ndarray[cnp.double_t, ndim=1] b):
    """
    Multiply a :class:`LLSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`LLSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        ``c = A * b``: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is more general as it takes into account strides in the numpy arrays and if the :class:`LLSparseMatrix`
        is symmetric or not.

    """
    # TODO: test, test, test!!!
    cdef INT_t A_nrow = A.nrow
    cdef INT_t A_ncol = A.ncol

    cdef size_t sd = sizeof(FLOAT_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    #cdef FLOAT_t * b_data = <FLOAT_t *> b.data

    cdef FLOAT_t * b_data = <FLOAT_t *> cnp.PyArray_DATA(b)


    # array c = A * b
    cdef cnp.ndarray[cnp.double_t, ndim=1] c = np.empty(A_nrow, dtype=np.float64)
    #cdef FLOAT_t * c_data = <FLOAT_t *> c.data
    cdef FLOAT_t * c_data = <FLOAT_t *> cnp.PyArray_DATA(c)


    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.is_symmetric:
            multiply_sym_ll_mat_with_numpy_vector_kernel(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
        else:
            multiply_ll_mat_with_numpy_vector_kernel(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
    else:
        if A.is_symmetric:
            multiply_sym_ll_mat_with_strided_numpy_vector_kernel(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.link, A.root)
        else:
            multiply_ll_mat_with_strided_numpy_vector_kernel(A.nrow,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.col, A.link, A.root)

    return c
