"""
Several helper routines to multiply an :class:`CSRSparseMatrix` with other matrices.

Implemented: :class:`CSRSparseMatrix` by

-  :program:`NumPy` vector:

    - ``A * b``
    - ``A^t * b``
    - ``A^h * b``
    - ``conj(A) * b``

- :class:`CSCSparseMatrix`



"""

###################################################
# CSRSparseMatrix by Numpy vector
###################################################
######################
# A * b
######################
cdef cnp.ndarray[cnp.npy_float32, ndim=1, mode='c'] multiply_csr_mat_with_numpy_vector_INT32_t_FLOAT32_t(CSRSparseMatrix_INT32_t_FLOAT32_t A, cnp.ndarray[cnp.npy_float32, ndim=1] b):
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

    cdef size_t sd = sizeof(FLOAT32_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef FLOAT32_t * b_data = <FLOAT32_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)

    cdef cnp.ndarray[cnp.npy_float32, ndim=1] c = np.empty(A_nrow, dtype=np.float32)
    cdef FLOAT32_t * c_data = <FLOAT32_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__store_symmetric:
            pass
            multiply_sym_csr_mat_with_numpy_vector_kernel_INT32_t_FLOAT32_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_csr_mat_with_numpy_vector_kernel_INT32_t_FLOAT32_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
    else:
        if A.__store_symmetric:
            multiply_sym_csr_mat_with_strided_numpy_vector_kernel_INT32_t_FLOAT32_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_csr_mat_with_strided_numpy_vector_kernel_INT32_t_FLOAT32_t(A.nrow,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.col, A.ind)

    return c


######################
# A^t * b
######################
cdef cnp.ndarray[cnp.npy_float32, ndim=1, mode='c'] multiply_transposed_csr_mat_with_numpy_vector_INT32_t_FLOAT32_t(CSRSparseMatrix_INT32_t_FLOAT32_t A, cnp.ndarray[cnp.npy_float32, ndim=1] b):
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

    cdef size_t sd = sizeof(FLOAT32_t)

    # test dimensions
    if A_nrow != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_ncol, A_nrow, b.size, 1))

    # direct access to vector b
    cdef FLOAT32_t * b_data = <FLOAT32_t *> cnp.PyArray_DATA(b)

    # array c = A^t * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)
    cdef cnp.ndarray[cnp.npy_float32, ndim=1] c = np.empty(A_ncol, dtype=np.float32)
    cdef FLOAT32_t * c_data = <FLOAT32_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__store_symmetric:
            multiply_sym_csr_mat_with_numpy_vector_kernel_INT32_t_FLOAT32_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_tranposed_csr_mat_with_numpy_vector_kernel_INT32_t_FLOAT32_t(A_nrow, A_ncol, b_data, c_data,
         A.val, A.col, A.ind)
    else:
        if A.__store_symmetric:
            multiply_sym_csr_mat_with_strided_numpy_vector_kernel_INT32_t_FLOAT32_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_tranposed_csr_mat_with_strided_numpy_vector_kernel_INT32_t_FLOAT32_t(A_nrow, A_ncol,
                                                                                      b_data, b.strides[0] / sd,
                                                                                      c_data, c.strides[0] / sd,
                                                                                      A.val, A.col, A.ind)

    return c



###################################################
# CSRSparseMatrix by a 2d matrix
###################################################
######################
# CSR by CSC
######################
cdef LLSparseMatrix_INT32_t_FLOAT32_t multiply_csr_mat_by_csc_mat_INT32_t_FLOAT32_t(CSRSparseMatrix_INT32_t_FLOAT32_t A, CSCSparseMatrix_INT32_t_FLOAT32_t B):

    # TODO: take into account if matrix A or B has its column indices ordered or not...
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
    # TODO: what strategy to implement?
    cdef INT32_t size_hint = A.nnz

    # TODO: maybe use MakeLLSparseMatrix and fix circular dependencies...
    C = LLSparseMatrix_INT32_t_FLOAT32_t(control_object=unexposed_value, nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, store_zero=store_zero)

    # CASES
    if not A.__store_symmetric and not B.__store_symmetric:
        pass
    else:
        raise NotImplemented("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    # TODO: what do we do? Column indices are NOT necessarily sorted...
    cdef:
        INT32_t i, j, k
        FLOAT32_t sum

    # don't keep zeros, no matter what
    cdef bint old_store_zero = store_zero
    C.__store_zero = 0

    for i from 0 <= i < C_nrow:
        for j from 0 <= j < C_ncol:

            sum = 0.0


            for k from 0 <= k < A_ncol:
                sum += (A[i, k] * B[k, j])

            C.put(i, j, sum)

    C.__store_zero = old_store_zero

    return C