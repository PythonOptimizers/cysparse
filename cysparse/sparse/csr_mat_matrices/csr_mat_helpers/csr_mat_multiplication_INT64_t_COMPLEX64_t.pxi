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
cdef cnp.ndarray[cnp.npy_complex64, ndim=1, mode='c'] multiply_csr_mat_with_numpy_vector_INT64_t_COMPLEX64_t(CSRSparseMatrix_INT64_t_COMPLEX64_t A, cnp.ndarray[cnp.npy_complex64, ndim=1] b):
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
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef size_t sd = sizeof(COMPLEX64_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef COMPLEX64_t * b_data = <COMPLEX64_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)

    cdef cnp.ndarray[cnp.npy_complex64, ndim=1] c = np.empty(A_nrow, dtype=np.complex64)
    cdef COMPLEX64_t * c_data = <COMPLEX64_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__use_symmetric_storage:
            pass
            multiply_sym_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
    else:
        if A.__use_symmetric_storage:
            multiply_sym_csr_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX64_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_csr_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX64_t(A.nrow,
                                                             b_data, b.strides[0] / sd,
                                                             c_data, c.strides[0] / sd,
                                                             A.val, A.col, A.ind)

    return c


######################
# A^t * b
######################
cdef cnp.ndarray[cnp.npy_complex64, ndim=1, mode='c'] multiply_transposed_csr_mat_with_numpy_vector_INT64_t_COMPLEX64_t(CSRSparseMatrix_INT64_t_COMPLEX64_t A, cnp.ndarray[cnp.npy_complex64, ndim=1] b):
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
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef size_t sd = sizeof(COMPLEX64_t)

    # test dimensions
    if A_nrow != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_ncol, A_nrow, b.size, 1))

    # direct access to vector b
    cdef COMPLEX64_t * b_data = <COMPLEX64_t *> cnp.PyArray_DATA(b)

    # array c = A^t * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)
    cdef cnp.ndarray[cnp.npy_complex64, ndim=1] c = np.empty(A_ncol, dtype=np.complex64)
    cdef COMPLEX64_t * c_data = <COMPLEX64_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__use_symmetric_storage:
            multiply_sym_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_tranposed_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, A_ncol, b_data, c_data,
         A.val, A.col, A.ind)
    else:
        if A.__use_symmetric_storage:
            multiply_sym_csr_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX64_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_tranposed_csr_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, A_ncol,
                                                                                      b_data, b.strides[0] / sd,
                                                                                      c_data, c.strides[0] / sd,
                                                                                      A.val, A.col, A.ind)

    return c


######################
# A^h * b
######################
cdef cnp.ndarray[cnp.npy_complex64, ndim=1, mode='c'] multiply_conjugate_transposed_csr_mat_with_numpy_vector_INT64_t_COMPLEX64_t(CSRSparseMatrix_INT64_t_COMPLEX64_t A, cnp.ndarray[cnp.npy_complex64, ndim=1] b):
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
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef size_t sd = sizeof(COMPLEX64_t)

    # test dimensions
    if A_nrow != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_ncol, A_nrow, b.size, 1))

    # direct access to vector b
    cdef COMPLEX64_t * b_data = <COMPLEX64_t *> cnp.PyArray_DATA(b)

    # array c = A^h * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)
    cdef cnp.ndarray[cnp.npy_complex64, ndim=1] c = np.empty(A_ncol, dtype=np.complex64)
    cdef COMPLEX64_t * c_data = <COMPLEX64_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__use_symmetric_storage:
            multiply_conjugate_transposed_sym_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, A_ncol, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_conjugate_transposed_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, A_ncol, b_data, c_data,
         A.val, A.col, A.ind)
    else:
        if A.__use_symmetric_storage:
            multiply_conjugate_transposed_sym_csr_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX64_t(A.nrow, A.ncol,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_conjugate_tranposed_csr_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, A_ncol,
                                                                                      b_data, b.strides[0] / sd,
                                                                                      c_data, c.strides[0] / sd,
                                                                                      A.val, A.col, A.ind)

    return c

######################
# conj(A) * b
######################
cdef cnp.ndarray[cnp.npy_complex64, ndim=1, mode='c'] multiply_conjugated_csr_mat_with_numpy_vector_INT64_t_COMPLEX64_t(CSRSparseMatrix_INT64_t_COMPLEX64_t A, cnp.ndarray[cnp.npy_complex64, ndim=1] b):
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
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef size_t sd = sizeof(COMPLEX64_t)

    # test dimensions
    if A_ncol != b.size:
        raise IndexError("Dimensions must agree ([%d,%d] * [%d, %d])" % (A_nrow, A_ncol, b.size, 1))

    # direct access to vector b
    cdef COMPLEX64_t * b_data = <COMPLEX64_t *> cnp.PyArray_DATA(b)

    # array c = A * b
    # TODO: check if we can not use static version of empty (cnp.empty instead of np.empty)

    cdef cnp.ndarray[cnp.npy_complex64, ndim=1] c = np.empty(A_nrow, dtype=np.complex64)
    cdef COMPLEX64_t * c_data = <COMPLEX64_t *> cnp.PyArray_DATA(c)

    # test if b vector is C-contiguous or not
    if cnp.PyArray_ISCONTIGUOUS(b):
        if A.__use_symmetric_storage:
            multiply_conjugated_sym_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)
        else:
            multiply_conjugated_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX64_t(A_nrow, b_data, c_data, A.val, A.col, A.ind)

    else:
        if A.__use_symmetric_storage:
            multiply_conjugated_sym_csr_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX64_t(A.nrow,
                                                                 b_data, b.strides[0] / sd,
                                                                 c_data, c.strides[0] / sd,
                                                                 A.val, A.col, A.ind)
        else:
            multiply_conjugated_csr_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX64_t(A.nrow,
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
cdef LLSparseMatrix_INT64_t_COMPLEX64_t multiply_csr_mat_by_csc_mat_INT64_t_COMPLEX64_t(CSRSparseMatrix_INT64_t_COMPLEX64_t A, CSCSparseMatrix_INT64_t_COMPLEX64_t B):

    # TODO: take into account if matrix A or B has its column indices ordered or not...
    # test dimensions
    cdef INT64_t A_nrow = A.nrow
    cdef INT64_t A_ncol = A.ncol

    cdef INT64_t B_nrow = B.nrow
    cdef INT64_t B_ncol = B.ncol

    if A_ncol != B_nrow:
        raise IndexError("Matrix dimensions must agree ([%d, %d] * [%d, %d])" % (A_nrow, A_ncol, B_nrow, B_ncol))

    cdef INT64_t C_nrow = A_nrow
    cdef INT64_t C_ncol = B_ncol

    cdef bint use_nonzero_storage = A.use_nonzero_storage and B.use_nonzero_storage
    # TODO: what strategy to implement?
    cdef INT64_t size_hint = A.nnz

    # TODO: maybe use MakeLLSparseMatrix and fix circular dependencies...
    C = LLSparseMatrix_INT64_t_COMPLEX64_t(control_object=unexposed_value, nrow=C_nrow, ncol=C_ncol, size_hint=size_hint, use_nonzero_storage=use_nonzero_storage)

    # CASES
    if not A.__use_symmetric_storage and not B.__use_symmetric_storage:
        pass
    else:
        raise NotImplemented("Multiplication with symmetric matrices is not implemented yet")

    # NON OPTIMIZED MULTIPLICATION
    # TODO: what do we do? Column indices are NOT necessarily sorted...
    cdef:
        INT64_t i, j, k
        COMPLEX64_t sum

    # don't keep zeros, no matter what
    cdef bint old_use_nonzero_storage = use_nonzero_storage
    C.__use_nonzero_storage = 0

    for i from 0 <= i < C_nrow:
        for j from 0 <= j < C_ncol:

            sum = 0.0 + 0.0j


            for k from 0 <= k < A_ncol:
                sum += (A[i, k] * B[k, j])

            C.put(i, j, sum)

    C.__use_nonzero_storage = old_use_nonzero_storage

    return C