


########################################################################################################################
# A * b
########################################################################################################################

###########################################
# C-contiguous, no symmetric
###########################################
cdef void multiply_csc_mat_with_numpy_vector_kernel_INT64_t_INT32_t(INT64_t m, INT64_t n, INT32_t *x, INT32_t *y,
         INT32_t *val, INT64_t *row, INT64_t *ind):
    """
    Compute ``y = A * x``.

    ``A`` is a :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t s
        INT64_t i, j, k

    # init numpy array
    for i from 0 <= i < m:

        y[i] = <INT32_t>0.0


    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:
            i = row[k]
            y[i] += val[k] * x[i]
