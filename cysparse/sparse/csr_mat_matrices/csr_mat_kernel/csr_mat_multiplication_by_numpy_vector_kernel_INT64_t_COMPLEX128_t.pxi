"""
Several routines to multiply a :class:`CSRSparseMatrix` matrix with a (comming from a :program:`NumPy` vector) C-array.

Covered cases:

1. :math:`A * b`:

- :program:`NumPy` array data C-contiguous, ``CSRSparseMatrix`` not symmetric
- :program:`NumPy` array data C-contiguous, ``CSRSparseMatrix`` symmetric
- :program:`NumPy` array data not C-contiguous, ``CSRSparseMatrix`` not symmetric
- :program:`NumPy` array data not C-contiguous, ``CSRSparseMatrix`` symmetric

2. :math:`A^t * b`

- :program:`NumPy` array data C-contiguous, ``CSRSparseMatrix`` not symmetric
- :program:`NumPy` array data not C-contiguous, ``CSRSparseMatrix`` not symmetric

Note:
    We only consider C-arrays with same type of elements as the type of elements in the ``CSRSparseMatrix``.
    Even if we construct the resulting :program:`NumPy` array as C-contiguous, the functions are more general and could
    be used with a given strided :program:`NumPy` `y` vector.
"""

########################################################################################################################
# A * b
########################################################################################################################

###########################################
# C-contiguous, non symmetric
###########################################
cdef void multiply_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, COMPLEX128_t *x, COMPLEX128_t *y,
         COMPLEX128_t *val, INT64_t *col, INT64_t *ind):
    """
    Compute ``y = A * x``.

    ``A`` is a :class:`CSRSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        COMPLEX128_t s
        INT64_t i, j, k

    for i from 0 <= i < m:

        s = <COMPLEX128_t>(0.0+0.0j)

        for k from ind[i] <= k < ind[i+1]:
            s += val[k] * x[col[k]]

        y[i] = s


###########################################
# C-contiguous, symmetric
###########################################
cdef void multiply_sym_csr_mat_with_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, COMPLEX128_t *x, COMPLEX128_t *y,
             COMPLEX128_t *val, INT64_t *col, INT64_t *ind):
    """
    Compute ``y = A * x``.

    ``A`` is a **symmetric** :class:`CSRSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT64_t i, j, k

    # init numpy array
    for i from 0 <= i < m:

        y[i] = <COMPLEX128_t>(0.0+0.0j)


    for i from 0 <= i < m:
        for k from ind[i] <= k < ind[i+1]:
            j = col[k]
            y[i] += val[k] * x[j]
            if i != j:
                y[j] += val[k] * x[i]
