


########################################################################################################################
# A * b
########################################################################################################################

###########################################
# C-contiguous, no symmetric
###########################################
cdef void multiply_csc_mat_with_numpy_vector_kernel_INT32_t_FLOAT64_t(INT32_t m, INT32_t n, FLOAT64_t *x, FLOAT64_t *y,
         FLOAT64_t *val, INT32_t *row, INT32_t *ind):
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
        FLOAT64_t s
        INT32_t i, j, k

    # init numpy array
    for i from 0 <= i < m:

        y[i] = <FLOAT64_t>0.0


    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:
            i = row[k]
            y[i] += val[k] * x[i]
