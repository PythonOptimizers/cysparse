###################################################
# CSRSparseMatrix by Numpy vector
###################################################
######################
# A * b
######################
cdef cnp.ndarray[cnp.npy_complex256, ndim=1, mode='c'] multiply_csr_mat_with_numpy_vector_INT32_t_COMPLEX256_t(CSRSparseMatrix_INT32_t_COMPLEX256_t A, cnp.ndarray[cnp.npy_complex256, ndim=1] b):
    """
    Multiply a :class:`CSRSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`CSRSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        ``c = A * b``: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is general as it takes into account strides in the numpy arrays and if the :class:`CSRSparseMatrix`
        is symmetric or not.


    """
    # TODO: test, test, test!!!
    cdef INT32_t A_nrow = A.nrow
    cdef INT32_t A_ncol = A.ncol

    cdef size_t sd = sizeof(COMPLEX256_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef COMPLEX256_t * b_data = <COMPLEX256_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)

    cdef cnp.ndarray[cnp.npy_complex256, ndim=1] c = np.empty(A_nrow, dtype=np.complex256)
    cdef COMPLEX256_t * c_data = <COMPLEX256_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__is_symmetric:
            pass
            multiply_sym_csr_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_csr_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
    else:
        if A.__is_symmetric:
            multiply_sym_csr_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_csr_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(A.nrow,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.col, A.ind)

    return c


######################
# A^t * b
######################
cdef cnp.ndarray[cnp.npy_complex256, ndim=1, mode='c'] multiply_transposed_csr_mat_with_numpy_vector_INT32_t_COMPLEX256_t(CSRSparseMatrix_INT32_t_COMPLEX256_t A, cnp.ndarray[cnp.npy_complex256, ndim=1] b):
    """
    Multiply a transposed :class:`CSRSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`CSRSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        :math:`c = A^t * b`: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is general as it takes into account strides in the numpy arrays and if the :class:`CSRSparseMatrix`
        is symmetric or not.

    """
    # TODO: test, test, test!!!
    cdef INT32_t A_nrow = A.nrow
    cdef INT32_t A_ncol = A.ncol

    cdef size_t sd = sizeof(COMPLEX256_t)

    # test dimensions
    if A_nrow != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_ncol, A_nrow, b.size, 1))

    # direct access to vector b
    cdef COMPLEX256_t * b_data = <COMPLEX256_t *> cnp.PyArray_DATA(b)

    # array c = A^t * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)
    cdef cnp.ndarray[cnp.npy_complex256, ndim=1] c = np.empty(A_ncol, dtype=np.complex256)
    cdef COMPLEX256_t * c_data = <COMPLEX256_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__is_symmetric:
            multiply_sym_csr_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
        else:
            print("Contiguous non symmetric")
            multiply_tranposed_csr_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, A_ncol, b_data, c_data,
         A.val, A.col, A.ind)
    else:
        if A.__is_symmetric:
            multiply_sym_csr_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_tranposed_csr_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, A_ncol,
                                                                                      b_data, b.strides[0] / sd,
                                                                                      c_data, c.strides[0] / sd,
                                                                                      A.val, A.col, A.ind)

    return c


######################
# A^h * b
######################
cdef cnp.ndarray[cnp.npy_complex256, ndim=1, mode='c'] multiply_conjugate_transposed_csr_mat_with_numpy_vector_INT32_t_COMPLEX256_t(CSRSparseMatrix_INT32_t_COMPLEX256_t A, cnp.ndarray[cnp.npy_complex256, ndim=1] b):
    """
    Multiply a conjugate transposed of a :class:`CSRSparseMatrix` ``A`` matrix with a numpy vector ``b``.

    Args
        A: A :class:`CSRSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        :math:`c = A^h * b`: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is general as it takes into account strides in the numpy arrays and if the :class:`CSRSparseMatrix`
        is symmetric or not.

    """
    # TODO: test, test, test!!!
    cdef INT32_t A_nrow = A.nrow
    cdef INT32_t A_ncol = A.ncol

    cdef size_t sd = sizeof(COMPLEX256_t)

    # test dimensions
    if A_nrow != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_ncol, A_nrow, b.size, 1))

    # direct access to vector b
    cdef COMPLEX256_t * b_data = <COMPLEX256_t *> cnp.PyArray_DATA(b)

    # array c = A^h * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)
    cdef cnp.ndarray[cnp.npy_complex256, ndim=1] c = np.empty(A_ncol, dtype=np.complex256)
    cdef COMPLEX256_t * c_data = <COMPLEX256_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__is_symmetric:
            multiply_conjugate_transposed_sym_csr_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, A_ncol, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_conjugate_transposed_csr_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, A_ncol, b_data, c_data,
         A.val, A.col, A.ind)
    else:
        if A.__is_symmetric:
            multiply_conjugate_transposed_sym_csr_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(A.nrow, A.ncol,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_conjugate_tranposed_csr_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, A_ncol,
                                                                                      b_data, b.strides[0] / sd,
                                                                                      c_data, c.strides[0] / sd,
                                                                                      A.val, A.col, A.ind)

    return c

######################
# conj(A) * b
######################
cdef cnp.ndarray[cnp.npy_complex256, ndim=1, mode='c'] multiply_conjugated_csr_mat_with_numpy_vector_INT32_t_COMPLEX256_t(CSRSparseMatrix_INT32_t_COMPLEX256_t A, cnp.ndarray[cnp.npy_complex256, ndim=1] b):
    """
    Multiply a conjugated :class:`CSRSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`CSRSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        ``c = conj(A) * b``: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is general as it takes into account strides in the numpy arrays and if the :class:`CSRSparseMatrix`
        is symmetric or not.


    """
    # TODO: test, test, test!!!
    cdef INT32_t A_nrow = A.nrow
    cdef INT32_t A_ncol = A.ncol

    cdef size_t sd = sizeof(COMPLEX256_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef COMPLEX256_t * b_data = <COMPLEX256_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)

    cdef cnp.ndarray[cnp.npy_complex256, ndim=1] c = np.empty(A_nrow, dtype=np.complex256)
    cdef COMPLEX256_t * c_data = <COMPLEX256_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__is_symmetric:
            raise NotImplementedError("eh...")
            multiply_conjugated_sym_csr_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_conjugated_csr_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)

    else:
        if A.__is_symmetric:
            raise NotImplementedError("eh...")
            multiply_conjugated_sym_csr_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_conjugated_csr_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(A.nrow,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.col, A.ind)

    return c


