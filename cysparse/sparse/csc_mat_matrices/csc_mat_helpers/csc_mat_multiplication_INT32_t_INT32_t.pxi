

###################################################
# CSCSparseMatrix by Numpy vector
###################################################
######################
# A * b
######################
cdef cnp.ndarray[cnp.npy_int32, ndim=1, mode='c'] multiply_csc_mat_with_numpy_vector_INT32_t_INT32_t(CSCSparseMatrix_INT32_t_INT32_t A, cnp.ndarray[cnp.npy_int32, ndim=1] b):
    """
    Multiply a :class:`CSCSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`CSCSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        ``c = A * b``: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is general as it takes into account strides in the numpy arrays and if the :class:`CSCSparseMatrix`
        is symmetric or not.


    """
    # TODO: test, test, test!!!
    cdef INT32_t A_nrow = A.nrow
    cdef INT32_t A_ncol = A.ncol

    cdef size_t sd = sizeof(INT32_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef INT32_t * b_data = <INT32_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)

    cdef cnp.ndarray[cnp.npy_int32, ndim=1] c = np.empty(A_nrow, dtype=np.int32)
    cdef INT32_t * c_data = <INT32_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__is_symmetric:
            pass
            multiply_sym_csc_mat_with_numpy_vector_kernel_INT32_t_INT32_t(A_nrow, A_ncol, b_data, c_data, A.val, A.row, A.ind)
        else:
            multiply_csc_mat_with_numpy_vector_kernel_INT32_t_INT32_t(A_nrow, A_ncol, b_data, c_data, A.val, A.row, A.ind)
    else:
        if A.__is_symmetric:
            multiply_sym_csc_mat_with_strided_numpy_vector_kernel_INT32_t_INT32_t(A.nrow, A.ncol,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.row, A.ind)
        else:
            multiply_csc_mat_with_strided_numpy_vector_kernel_INT32_t_INT32_t(A.nrow, A.ncol,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.row, A.ind)

    return c


######################
# A^t * b
######################
cdef cnp.ndarray[cnp.npy_int32, ndim=1, mode='c'] multiply_transposed_csc_mat_with_numpy_vector_INT32_t_INT32_t(CSCSparseMatrix_INT32_t_INT32_t A, cnp.ndarray[cnp.npy_int32, ndim=1] b):
    """
    Multiply a transposed :class:`CSCSparseMatrix` ``A`` with a numpy vector ``b``.

    Args
        A: A :class:`CSCSparseMatrix`.
        b: A numpy.ndarray of dimension 1 (a vector).

    Returns:
        :math:`c = A^t * b`: a **new** numpy.ndarray of dimension 1.

    Raises:
        IndexError if dimensions don't match.

    Note:
        This version is general as it takes into account strides in the numpy arrays and if the :class:`CSCSparseMatrix`
        is symmetric or not.

    """
    # TODO: test, test, test!!!
    cdef INT32_t A_nrow = A.nrow
    cdef INT32_t A_ncol = A.ncol

    cdef size_t sd = sizeof(INT32_t)

    # test dimensions
    if A_nrow != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_ncol, A_nrow, b.size, 1))

    # direct access to vector b
    cdef INT32_t * b_data = <INT32_t *> cnp.PyArray_DATA(b)

    # array c = A^t * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)
    cdef cnp.ndarray[cnp.npy_int32, ndim=1] c = np.empty(A_ncol, dtype=np.int32)
    cdef INT32_t * c_data = <INT32_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__is_symmetric:
            multiply_sym_csc_mat_with_numpy_vector_kernel_INT32_t_INT32_t(A_nrow, A_ncol, b_data, c_data, A.val, A.row, A.ind)
        else:
            multiply_tranposed_csc_mat_with_numpy_vector_kernel_INT32_t_INT32_t(A_nrow, A_ncol, b_data, c_data,
         A.val, A.row, A.ind)
    else:
        if A.__is_symmetric:
            multiply_sym_csc_mat_with_strided_numpy_vector_kernel_INT32_t_INT32_t(A.nrow, A_ncol,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.row, A.ind)
        else:
            multiply_tranposed_csc_mat_with_strided_numpy_vector_kernel_INT32_t_INT32_t(A_nrow, A_ncol,
                                                                                      b_data, b.strides[0] / sd,
                                                                                      c_data, c.strides[0] / sd,
                                                                                      A.val, A.row, A.ind)

    return c