"""
Several helper routines for addition with/by a ``LLSparseMatrix`` matrix.
"""

########################################################################################################################
# Addition functions
########################################################################################################################

###################################################
# LLSparseMatrix by LLSparseMatrix
###################################################

cdef update_add_at_with_numpy_arraysINT64_t_FLOAT64_t(LLSparseMatrix_INT64_t_FLOAT64_t A,
                                                   cnp.ndarray[cnp.npy_int64, ndim=1, mode='c'] id1,
                                                   cnp.ndarray[cnp.npy_int64, ndim=1, mode='c'] id2,
                                                   cnp.ndarray[cnp.npy_float64, ndim=1] val):
    """
    Update of matrix in place by a vector.

    This operation is equivalent to

    ..  code-block:: python

        for i in range(len(val)):
            A[id1[i],id2[i]] += val[i]

    Warning:
        Index arrays **must** be C-contiguous.

        There is not of out of bounds test.
    """
    # test dimensions
    cdef Py_ssize_t array_size = id1.size
    cdef INT64_t i

    if array_size != id2.size or array_size != val.size:
        raise IndexError("Arrays dimensions must agree")

    # strided val?
    cdef size_t sd = sizeof(FLOAT64_t)
    cdef INT64_t incx = <INT64_t> val.strides[0] / sd

    # direct access to arrays
    cdef FLOAT64_t * val_data = <FLOAT64_t *> cnp.PyArray_DATA(val)
    cdef INT64_t * id1_data = <INT64_t *> cnp.PyArray_DATA(id1)
    cdef INT64_t * id2_data = <INT64_t *> cnp.PyArray_DATA(id2)

    if cnp.PyArray_ISCONTIGUOUS(val):
        for i from 0<= i < array_size:
            update_ll_mat_item_add_INT64_t_FLOAT64_t(A, id1_data[i], id2_data[i], val_data[i])
    else:
        # val is not C-contiguous
        for i from 0<= i < array_size:
            update_ll_mat_item_add_INT64_t_FLOAT64_t(A, id1_data[i], id2_data[i], val_data[i * incx])
