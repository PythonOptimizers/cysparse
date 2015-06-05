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

    """
    pass