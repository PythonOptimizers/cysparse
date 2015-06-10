

###################################################
# CSCSparseMatrix by Numpy vector
###################################################
######################
# A * b
######################
cdef cnp.ndarray[cnp.npy_int64, ndim=1, mode='c'] multiply_csc_mat_with_numpy_vector_INT64_t_INT64_t(CSCSparseMatrix_INT64_t_INT64_t A, cnp.ndarray[cnp.npy_int64, ndim=1] b):
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
            pass
            #multiply_sym_csc_mat_with_numpy_vector_kernel_INT64_t_INT64_t(A_nrow, b_data, c_data, A.val, A.row, A.ind)
        else:
            multiply_csc_mat_with_numpy_vector_kernel_INT64_t_INT64_t(A_nrow, A_ncol, b_data, c_data, A.val, A.row, A.ind)
    else:
        pass
        #if A.is_symmetric:
        #    multiply_sym_csc_mat_with_strided_numpy_vector_kernel_INT64_t_INT64_t(A.nrow,
        #                                                         b_data, b.strides[0] / sd,
        #                                                         c_data, c.strides[0] / sd,
        #                                                         A.val, A.row, A.ind)
        #else:
        #    multiply_csc_mat_with_strided_numpy_vector_kernel_INT64_t_INT64_t(A.nrow,
        #                                                     b_data, b.strides[0] / sd,
        #                                                     c_data, c.strides[0] / sd,
        #                                                     A.val, A.row, A.ind)

    return c