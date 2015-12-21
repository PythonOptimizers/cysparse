"""
Several helper routines for multiplication with/by a ``LLSparseMatrix`` matrix.

Covered cases:

1. ``LLSparseMatrix`` by another two dimensional matrix:

    - ``LLSparseMatrix`` by ``LLSparseMatrix``;
    - ``Transposed LLSparseMatrix`` by ``LLSparseMatrix``;
    - ``LLSparseMatrix`` by ``NumPy`` array;

2. ``LLSparseMatrix`` by ``Numpy`` vector

    - A * b;
    - A^t * b;
    - A^h * b;
    - conj(A) * b


"""

########################################################################################################################
# LLSparseMatrix by another two dimensional matrix
########################################################################################################################

###################################################
# LLSparseMatrix by LLSparseMatrix
###################################################
cdef LLSparseMatrix_INT32_t_FLOAT64_t multiply_two_ll_mat_INT32_t_FLOAT64_t(LLSparseMatrix_INT32_t_FLOAT64_t A, LLSparseMatrix_INT32_t_FLOAT64_t B):
    """
    Multiply two :class:`LLSparseMatrix_INT32_t_FLOAT64_t` ``A`` and ``B``.

    Args:
        A: An :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``A``.
        B: An :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``B``.

    Returns:
        A **new** :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``C = A * B``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplementedError``: When matrix ``A`` or ``B`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    # TODO: LLSparseMatrix * A, LLSparseMatrix * B ...
    # test dimensions
    cdef INT32_t A_nrow = A.nrow
    cdef INT32_t A_ncol = A.ncol

    cdef INT32_t B_nrow = B.nrow
    cdef INT32_t B_ncol = B.ncol

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT32_t C_nrow = A_nrow
    cdef INT32_t C_ncol = B_ncol

    cdef bint store_zero = A.store_zero and B.store_zero
    cdef INT32_t size_hint = A.size_hint

    C = LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zero=store_zero)


    # CASES
    if not A.__store_symmetric and not B.__store_symmetric:
        pass
    else:
        raise NotImplementedError("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    cdef:
        FLOAT64_t valA
        INT32_t iA, jA, kA, kB

    for iA from 0 <= iA < A_nrow:
        kA = A.root[iA]

        while kA != -1:
            valA = A.val[kA]
            jA = A.col[kA]
            kA = A.link[kA]

            # add jA-th row of B to iA-th row of C
            kB = B.root[jA]
            while kB != -1:
                update_ll_mat_item_add_INT32_t_FLOAT64_t(C, iA, B.col[kB], valA*B.val[kB])
                kB = B.link[kB]
    return C


###################################################
# Transposed LLSparseMatrix by LLSparseMatrix
###################################################
cdef LLSparseMatrix_INT32_t_FLOAT64_t multiply_transposed_ll_mat_by_ll_mat_INT32_t_FLOAT64_t(LLSparseMatrix_INT32_t_FLOAT64_t A, LLSparseMatrix_INT32_t_FLOAT64_t B):
    """
    Multiply two :class:`LLSparseMatrix_INT32_t_FLOAT64_t` ``A`` and ``B``: :math:`A^t * B`.

    Args:
        A: An :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``A``.
        B: An :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``B``.

    Returns:
        A **new** :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``C = A^t * B``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplementedError``: When matrix ``A`` or ``B`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    # test dimensions
    cdef INT32_t A_nrow = A.__nrow
    cdef INT32_t A_ncol = A.__ncol

    cdef INT32_t B_nrow = B.__nrow
    cdef INT32_t B_ncol = B.__ncol

    if A_nrow != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d]^t * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT32_t C_nrow = A_ncol
    cdef INT32_t C_ncol = B_ncol

    cdef bint store_zero = A.store_zero and B.store_zero
    # TODO: is this a good idea?
    cdef INT32_t size_hint = min(A.__nnz, B.__nnz)

    C = LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zero=store_zero)

    # CASES
    if not A.__store_symmetric and not B.__store_symmetric:
        # we only deal with non symmetric matrices
        pass
    else:
        raise NotImplementedError("Multiplication with symmetric matrices is not implemented yet")

    cdef:
        FLOAT64_t valA
        INT32_t iA, kA, iC, kB, ret

    for iA from 0<= iA < A_nrow:
        kA = A.root[iA]

        while kA != -1:
            valA = A.val[kA]
            iC = A.col[kA]

            kB = B.root[iA]
            while kB != -1:
                update_ll_mat_item_add_INT32_t_FLOAT64_t(C, iC, B.col[kB], valA*B.val[kB])

                kB = B.link[kB]

            kA = A.link[kA]

    return C


###################################################
# LLSparseMatrix by a full NumPy matrix
###################################################
cdef cnp.ndarray[cnp.npy_float64, ndim=2] multiply_ll_mat_with_numpy_ndarray_FLOAT64_t(LLSparseMatrix_INT32_t_FLOAT64_t A, cnp.ndarray[cnp.npy_float64, ndim=2] B):
    """
    Multiply a :class:`LLSparseMatrix_INT32_t_FLOAT64_t` ``A`` with a dense :program:`NumPy` ``B``.

    Args:
        A: An :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``A``.
        B: An :program:`NumPy` ``B``.

    Returns:
        A **new** :program:`NumPy` ``C = A * B``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplementedError``: When matrix ``A`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    # test dimensions
    cdef INT32_t A_nrow = A.__nrow
    cdef INT32_t A_ncol = A.__ncol

    cdef INT32_t B_nrow, B_ncol
    B_nrow = B.shape[0]
    B_ncol = B.shape[1]

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT32_t C_nrow = A_nrow
    cdef INT32_t C_ncol = B_ncol

    cdef cnp.ndarray[cnp.npy_float64, ndim=2] C = np.zeros((C_nrow,C_ncol), dtype=np.float64)

    # memory views
    cdef FLOAT64_t [:, :] B_memory_view = B
    cdef FLOAT64_t [:, :] C_memory_view = C

    # NON OPTIMIZED MULTIPLICATION
    # memory views are not really optimized either...
    cdef:
        FLOAT64_t valA
        INT32_t iA, jA, kA, jB

    # CASES
    if not A.__store_symmetric:
        for iA from 0 <= iA < A_nrow:
            kA = A.root[iA]

            while kA != -1:
                valA = A.val[kA]
                jA = A.col[kA]

                for jB from 0 <= jB < B_ncol:
                    C_memory_view[iA, jB] += valA * B_memory_view[jA, jB]

                kA = A.link[kA]
    else:
        for iA from 0 <= iA < A_nrow:
            kA = A.root[iA]

            while kA != -1:
                valA = A.val[kA]
                jA = A.col[kA]

                for jB from 0 <= jB < B_ncol:
                    C_memory_view[iA, jB] += valA * B_memory_view[jA, jB]

                if jA != iA:
                    for jB from 0 <= jB < B_ncol:
                        C_memory_view[jA, jB] += valA * B_memory_view[iA, jB]

                kA = A.link[kA]

    return C

cdef LLSparseMatrix_INT32_t_FLOAT64_t multiply_transposed_ll_mat_with_self(LLSparseMatrix_INT32_t_FLOAT64_t A):
    """
    Multiply a :class:`LLSparseMatrix_INT32_t_FLOAT64_t` ``A`` with itself.

    Args:
        A: An :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``A``.

    Returns:
        A **new** ``LLSparseMatrix_INT32_t_FLOAT64_t`` ``C = A^t * A``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplementedError``: When matrix ``A`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    if A.store_symmetric:
        raise NotImplementedError('matdot_transp_self peration with symmetric matrices not supported')

    cdef:
        bint store_zero = A.store_zero
        INT32_t size_hint = A.size_hint
        LLSparseMatrix_INT32_t_FLOAT64_t C
        INT32_t iA, iC, kA, kA2
        FLOAT64_t valA

    C = LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=A.ncol, ncol=A.ncol, size_hint=size_hint, store_zero=store_zero, store_symmetric=True)

    for iA from 0 <= iA < A.nrow:
        kA = A.root[iA]
        while kA != -1:
            iC = A.col[kA]
            valA = A.val[kA]

            kA2 = A.root[iA]
            while kA2 != -1:
                if iC >= A.col[kA2]:
                    update_ll_mat_item_add_INT32_t_FLOAT64_t(C, iC, A.col[kA2], valA * A.val[kA2])

                kA2 = A.link[kA2]

            kA = A.link[kA]

    return C

cdef LLSparseMatrix_INT32_t_FLOAT64_t multiply_transposed_ll_mat_with_self_scaled(LLSparseMatrix_INT32_t_FLOAT64_t A, cnp.ndarray[cnp.npy_float64, ndim=1] d):
    """
    Multiply a :class:`LLSparseMatrix_INT32_t_FLOAT64_t` ``A`` with itself and a scale vector.

    Args:
        A: An :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``A``.
        b: A :program:`NumPy` scaling vector.

    Returns:
        A **new** ``LLSparseMatrix_INT32_t_FLOAT64_t`` ``C = A^t * D * A`` where ``D=diag(d)``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplementedError``: When matrix ``A`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    if A.store_symmetric:
        raise NotImplementedError('matdot_transp_self peration with symmetric matrices not supported')

    cdef:
        bint store_zero = A.store_zero
        INT32_t size_hint = A.size_hint
        LLSparseMatrix_INT32_t_FLOAT64_t C
        INT32_t iA, iC, kA, kA2
        FLOAT64_t valA

    # direct access to vector d
    cdef FLOAT64_t * d_data = <FLOAT64_t *> cnp.PyArray_DATA(d)

    # stride if any
    cdef:
        size_t sd = sizeof(FLOAT64_t)
        INT32_t incx = d.strides[0] / sd


    C = LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=A.ncol, ncol=A.ncol, size_hint=size_hint, store_zero=store_zero, store_symmetric=True)

    if cnp.PyArray_ISCONTIGUOUS(d):
        for iA from 0 <= iA < A.nrow:
            kA = A.root[iA]
            while kA != -1:
                iC = A.col[kA]
                valA = A.val[kA] * d_data[iA]

                kA2 = A.root[iA]
                while kA2 != -1:
                    if iC >= A.col[kA2]:
                        update_ll_mat_item_add_INT32_t_FLOAT64_t(C, iC, A.col[kA2], valA * A.val[kA2])

                    kA2 = A.link[kA2]

                kA = A.link[kA]

    else:  #  d not C-contiguous
        for iA from 0 <= iA < A.nrow:
            kA = A.root[iA]
            while kA != -1:
                iC = A.col[kA]
                valA = A.val[kA] * d_data[iA * incx]

                kA2 = A.root[iA]
                while kA2 != -1:
                    if iC >= A.col[kA2]:
                        update_ll_mat_item_add_INT32_t_FLOAT64_t(C, iC, A.col[kA2], valA * A.val[kA2])

                    kA2 = A.link[kA2]

                kA = A.link[kA]

    return C

###################################################
# Transposed LLSparseMatrix by a full NumPy matrix
###################################################
cdef cnp.ndarray[cnp.npy_float64, ndim=2] multiply_transposed_ll_mat_with_numpy_ndarray_FLOAT64_t(LLSparseMatrix_INT32_t_FLOAT64_t A, cnp.ndarray[cnp.npy_float64, ndim=2] B):
    """
    Multiply a transposed :class:`LLSparseMatrix_INT32_t_FLOAT64_t` ``A`` with a dense :program:`NumPy` ``B``.

    Args:
        A: An :class:``LLSparseMatrix_INT32_t_FLOAT64_t`` ``A``.
        B: An :program:`NumPy` ``B``.

    Returns:
        A **new** :program:`NumPy` ``C = A^t * B``.

    Raises:
        ``IndexError`` if matrix dimension don't agree.
        ``NotImplementedError``: When matrix ``A`` is symmetric.
        ``RuntimeError`` if some error occurred during the computation.
    """
    # test dimensions
    cdef INT32_t A_nrow = A.__nrow
    cdef INT32_t A_ncol = A.__ncol

    cdef INT32_t B_nrow, B_ncol
    B_nrow = B.shape[0]
    B_ncol = B.shape[1]

    if A_nrow != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d]^t * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT32_t C_nrow = A_ncol
    cdef INT32_t C_ncol = B_ncol

    cdef cnp.ndarray[cnp.npy_float64, ndim=2] C = np.zeros((C_nrow,C_ncol), dtype=np.float64)

    # memory views
    cdef FLOAT64_t [:, :] B_memory_view = B
    cdef FLOAT64_t [:, :] C_memory_view = C

    # NON OPTIMIZED MULTIPLICATION
    # memory views are not really optimized either...
    cdef:
        FLOAT64_t valA
        INT32_t iA, jA, kA, jB

    # CASES
    if not A.__store_symmetric:
        for iA from 0 <= iA < A_nrow:
            kA = A.root[iA]

            while kA != -1:
                valA = A.val[kA]
                jA = A.col[kA]

                for jB from 0 <= jB < B_ncol:
                    C_memory_view[jA, jB] += valA * B_memory_view[iA, jB]

                kA = A.link[kA]
    else:
        # TODO: maybe uncomment following line for maitenance?
        # return multiply_ll_mat_with_numpy_ndarray_FLOAT64_t(A, B) # works of course but it is the same here
        for iA from 0 <= iA < A_nrow:
            kA = A.root[iA]

            while kA != -1:
                valA = A.val[kA]
                jA = A.col[kA]

                for jB from 0 <= jB < B_ncol:
                    C_memory_view[jA, jB] += valA * B_memory_view[iA, jB]

                if jA != iA:
                    for jB from 0 <= jB < B_ncol:
                        C_memory_view[iA, jB] += valA * B_memory_view[jA, jB]

                kA = A.link[kA]

    return C

########################################################################################################################
# LLSparseMatrix by Numpy vector
########################################################################################################################
######################
# A * b
######################
cdef cnp.ndarray[cnp.npy_float64, ndim=1, mode='c'] multiply_ll_mat_with_numpy_vector_INT32_t_FLOAT64_t(LLSparseMatrix_INT32_t_FLOAT64_t A, cnp.ndarray[cnp.npy_float64, ndim=1] b):
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
        This version is general as it takes into account strides in the numpy arrays and if the :class:`LLSparseMatrix`
        is symmetric or not.

    """
    # TODO: test, test, test!!!
    cdef INT32_t A_nrow = A.__nrow
    cdef INT32_t A_ncol = A.__ncol

    cdef size_t sd = sizeof(FLOAT64_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef FLOAT64_t * b_data = <FLOAT64_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)

    cdef cnp.ndarray[cnp.npy_float64, ndim=1] c = np.empty(A_nrow, dtype=np.float64)
    cdef FLOAT64_t * c_data = <FLOAT64_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__store_symmetric:
            multiply_sym_ll_mat_with_numpy_vector_kernel_INT32_t_FLOAT64_t(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
        else:
            multiply_ll_mat_with_numpy_vector_kernel_INT32_t_FLOAT64_t(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
    else:
        if A.__store_symmetric:
            multiply_sym_ll_mat_with_strided_numpy_vector_kernel_INT32_t_FLOAT64_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.link, A.root)
        else:
            multiply_ll_mat_with_strided_numpy_vector_kernel_INT32_t_FLOAT64_t(A.nrow,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.col, A.link, A.root)

    return c


######################
# A^t * b
######################
cdef cnp.ndarray[cnp.npy_float64, ndim=1, mode='c'] multiply_transposed_ll_mat_with_numpy_vector_INT32_t_FLOAT64_t(LLSparseMatrix_INT32_t_FLOAT64_t A, cnp.ndarray[cnp.npy_float64, ndim=1] b):
    """
    Multiply a transposed of a :class:`LLSparseMatrix` ``A`` matrix with a numpy vector ``b``.

    Args
        A: A :class:`LLSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        :math:`c = A^t * b`: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is general as it takes into account strides in the numpy arrays and if the :class:`LLSparseMatrix`
        is symmetric or not.

    """
    # TODO: test, test, test!!!
    cdef INT32_t A_nrow = A.__nrow
    cdef INT32_t A_ncol = A.__ncol

    cdef size_t sd = sizeof(FLOAT64_t)

    # test dimensions
    if A_nrow != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_ncol, A_nrow, b.size, 1))

    # direct access to vector b
    cdef FLOAT64_t * b_data = <FLOAT64_t *> cnp.PyArray_DATA(b)

    # array c = A^t * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)
    cdef cnp.ndarray[cnp.npy_float64, ndim=1] c = np.empty(A_ncol, dtype=np.float64)
    cdef FLOAT64_t * c_data = <FLOAT64_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__store_symmetric:
            multiply_sym_ll_mat_with_numpy_vector_kernel_INT32_t_FLOAT64_t(A_nrow, b_data, c_data, A.val, A.col, A.link, A.root)
        else:
            multiply_tranposed_ll_mat_with_numpy_vector_kernel_INT32_t_FLOAT64_t(A_nrow, A_ncol, b_data, c_data,
         A.val, A.col, A.link, A.root)
    else:
        if A.__store_symmetric:
            multiply_sym_ll_mat_with_strided_numpy_vector_kernel_INT32_t_FLOAT64_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.link, A.root)
        else:
            multiply_tranposed_ll_mat_with_strided_numpy_vector_kernel_INT32_t_FLOAT64_t(A_nrow, A_ncol,
                                                                                      b_data, b.strides[0] / sd,
                                                                                      c_data, c.strides[0] / sd,
                                                                                      A.val, A.col, A.link, A.root)

    return c

