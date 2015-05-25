"""
Several helper routines for multiplication with/by a ``LLSparseMatrix`` matrix.
"""

########################################################################################################################
# Multiplication functions
########################################################################################################################

###################################################
# LLSparseMatrix by LLSparseMatrix
###################################################
cdef LLSparseMatrix_INT64_t_INT64_t multiply_two_ll_mat_INT64_t_INT64_t(LLSparseMatrix_INT64_t_INT64_t A, LLSparseMatrix_INT64_t_INT64_t B):
    """
    Multiply two :class:`LLSparseMatrix_INT64_t_INT64_t` ``A`` and ``B``.

    Args:
        A: An :class:``LLSparseMatrix_INT64_t_INT64_t`` ``A``.
        B: An :class:``LLSparseMatrix_INT64_t_INT64_t`` ``B``.

    Returns:
        A **new** :class:``LLSparseMatrix_INT64_t_INT64_t`` ``C = A * B``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplementedError``: When matrix ``A`` or ``B`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    # TODO: LLSparseMatrix * A, LLSparseMatrix * B ...
    # test dimensions
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef INT64_t B_nrow = B.nrow
    cdef INT64_t B_ncol = B.ncol

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT64_t C_nrow = A_nrow
    cdef INT64_t C_ncol = B_ncol

    cdef bint store_zeros = A.store_zeros and B.store_zeros
    cdef INT64_t size_hint = A.size_hint

    C = LLSparseMatrix_INT64_t_INT64_t(control_object=unexposed_value, nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zeros=store_zeros)


    # CASES
    if not A.is_symmetric and not B.is_symmetric:
        pass
    else:
        raise NotImplementedError("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    cdef:
        INT64_t valA
        INT64_t iA, jA, kA, kB

    for iA from 0 <= iA < A_nrow:
        kA = A.root[iA]

        while kA != -1:
            valA = A.val[kA]
            jA = A.col[kA]
            kA = A.link[kA]

            # add jA-th row of B to iA-th row of C
            kB = B.root[jA]
            while kB != -1:
                update_ll_mat_item_add_INT64_t_INT64_t(C, iA, B.col[kB], valA*B.val[kB])
                kB = B.link[kB]
    return C


###################################################
# LLSparseMatrix by Numpy vector
###################################################
######################
# A * b
######################
cdef cnp.ndarray[cnp.npy_int64, ndim=1, mode='c'] multiply_ll_mat_with_numpy_vector_INT64_t_INT64_t(LLSparseMatrix_INT64_t_INT64_t A, cnp.ndarray[cnp.npy_int64, ndim=1] b):
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
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef size_t sd = sizeof(INT64_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef INT64_t * b_data = <INT64_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)

    cdef cnp.ndarray[cnp.npy_int64, ndim=1] c = np.empty(A_nrow, dtype=np.int64)
    cdef INT64_t * c_data = <INT64_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.is_symmetric:
            multiply_sym_ll_mat_with_numpy_vector_kernel_INT64_t_INT64_t(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
        else:
            multiply_ll_mat_with_numpy_vector_kernel_INT64_t_INT64_t(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
    else:
        if A.is_symmetric:
            multiply_sym_ll_mat_with_strided_numpy_vector_kernel_INT64_t_INT64_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.link, A.root)
        else:
            multiply_ll_mat_with_strided_numpy_vector_kernel_INT64_t_INT64_t(A.nrow,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.col, A.link, A.root)

    return c

######################
# A^t * b
######################
cdef cnp.ndarray[cnp.npy_int64, ndim=1, mode='c'] multiply_transposed_ll_mat_with_numpy_vector_INT64_t_INT64_t(LLSparseMatrix_INT64_t_INT64_t A, cnp.ndarray[cnp.npy_int64, ndim=1] b):
    """
    Multiply a transposed :class:`LLSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`LLSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        :math:`c = A^t * b`: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is more general as it takes into account strides in the numpy arrays and if the :class:`LLSparseMatrix`
        is symmetric or not.

    """
    # TODO: test, test, test!!!
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef size_t sd = sizeof(INT64_t)

    # test dimensions
    if A_nrow != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_ncol, A_nrow, b.size, 1))

    # direct access to vector b
    cdef INT64_t * b_data = <INT64_t *> cnp.PyArray_DATA(b)

    # array c = A^t * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)
    cdef cnp.ndarray[cnp.npy_int64, ndim=1] c = np.empty(A_ncol, dtype=np.int64)
    cdef INT64_t * c_data = <INT64_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.is_symmetric:
            multiply_sym_ll_mat_with_numpy_vector_kernel_INT64_t_INT64_t(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
        else:
            multiply_tranposed_ll_mat_with_numpy_vector_kernel_INT64_t_INT64_t(A_nrow, A_ncol, b_data, c_data,
         A.val, A.col, A.link, A.root)
    else:
        if A.is_symmetric:
            multiply_sym_ll_mat_with_strided_numpy_vector_kernel_INT64_t_INT64_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.link, A.root)
        else:
            multiply_tranposed_ll_mat_with_with_strided_numpy_vector_kernel_INT64_t_INT64_t(A_nrow, A_ncol,
                                                                                           b_data, b.strides[0] / sd,
                                                                                           c_data, c.strides[0] / sd,
                                                                                           A.val, A.col, A.link, A.root)

    return c