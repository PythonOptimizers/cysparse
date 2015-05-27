"""
Several routines to multiply a :class:`LLSparseMatrix` matrix with a (comming from a :program:`NumPy` vector) C-array.

Covered cases:

1. :math:`A * b`:

- :program:`NumPy` array data C-contiguous, ``LLSparseMatrix`` not symmetric
- :program:`NumPy` array data C-contiguous, ``LLSparseMatrix`` symmetric
- :program:`NumPy` array data not C-contiguous, ``LLSparseMatrix`` not symmetric
- :program:`NumPy` array data not C-contiguous, ``LLSparseMatrix`` symmetric

2. :math:`A^t * b`

- :program:`NumPy` array data C-contiguous, ``LLSparseMatrix`` not symmetric
- :program:`NumPy` array data not C-contiguous, ``LLSparseMatrix`` not symmetric

Note:
    We only consider C-arrays with same type of elements as the type of elements in the ``LLSparseMatrix``.
    Even if we construct the resulting :program:`NumPy` array as C-contiguous, the functions are more general and could
    be used with a given strided :program:`NumPy` `y` vector.
"""

########################################################################################################################
# A * b
########################################################################################################################

###########################################
# C-contiguous, no symmetric
###########################################
cdef void multiply_ll_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(INT32_t m, COMPLEX256_t *x, COMPLEX256_t *y,
         COMPLEX256_t *val, INT32_t *col, INT32_t *link, INT32_t *root):
    """
    Compute ``y = A * x``.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        COMPLEX256_t s
        INT32_t i, k

    for i from 0 <= i < m:

        s = <COMPLEX256_t>(0.0+0.0j)

        k = root[i]

        while k != -1:
          s += val[k] * x[col[k]]
          k = link[k]

        y[i] = s

###########################################
# C-contiguous, symmetric
###########################################
cdef void multiply_sym_ll_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(INT32_t m, COMPLEX256_t *x, COMPLEX256_t *y,
             COMPLEX256_t *val, INT32_t *col, INT32_t *link, INT32_t *root):
    """
    Compute ``y = A * x``.

    ``A`` is a **symmetric** :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        COMPLEX256_t s, v, xi
        INT32_t i, j, k

    for i from 0 <= i < m:
        xi = x[i]

        s = <COMPLEX256_t>(0.0+0.0j)

        k = root[i]

        while k != -1:
            j = col[k]
            v = val[k]
            s += v * x[j]
            if i != j:
                y[j] += v * xi
            k = link[k]

        y[i] = s

###########################################
# Non C-contiguous, non symmetric
###########################################
cdef void multiply_ll_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(INT32_t m,
            COMPLEX256_t *x, INT32_t incx,
            COMPLEX256_t *y, INT32_t incy,
            COMPLEX256_t *val, INT32_t *col, INT32_t *link, INT32_t *root):
    """
    Compute ``y = A * x``.

    ``A`` is :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional **non** C-contiguous numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider *both* numpy arrays as **non** C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        COMPLEX256_t s
        INT32_t i, k

    for i from 0 <= i < m:

        s = <COMPLEX256_t>(0.0+0.0j)

        k = root[i]

        while k != -1:
            s += val[k] * x[col[k]*incx]
            k = link[k]

        y[i*incy] = s

###########################################
# Non C-contiguous, symmetric
###########################################
cdef void multiply_sym_ll_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(INT32_t m,
                COMPLEX256_t *x, INT32_t incx,
                COMPLEX256_t *y, INT32_t incy,
                COMPLEX256_t *val, INT32_t *col, INT32_t *link, INT32_t *root):
    """
    Compute ``y = A * x``.

    ``A`` is a **symmetric** :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional **non** C-contiguous numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider *both* numpy arrays as **non** C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        COMPLEX256_t s, v, xi
        INT32_t i, j, k

    for i from 0 <= i < m:
        xi = x[i*incx]

        s = <COMPLEX256_t>(0.0+0.0j)

        k = root[i]

        while k != -1:
            j = col[k]
            v = val[k]
            s += v * x[j*incx]
            if i != j:
                y[j*incy] += v * xi
            k = link[k]

        y[i*incy] = s


########################################################################################################################
# A^t * b
########################################################################################################################

###########################################
# C-contiguous, no symmetric
###########################################
cdef void multiply_tranposed_ll_mat_with_numpy_vector_kernel_INT32_t_COMPLEX256_t(INT32_t m, INT32_t n, COMPLEX256_t *x, COMPLEX256_t *y,
         COMPLEX256_t *val, INT32_t *col, INT32_t *link, INT32_t *root):
    """
    Compute :math:`y = A^t * x`.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        COMPLEX256_t xi
        INT32_t i, k

    for i from 0 <= i < n:

        y[i] = <COMPLEX256_t>(0.0+0.0j)


    for i from 0 <= i < m:
        xi = x[i]
        k = root[i]

        while k != -1:
          y[col[k]] += val[k] * xi
          k = link[k]

###########################################
# Non C-contiguous, no symmetric
###########################################
cdef void multiply_tranposed_ll_mat_with_with_strided_numpy_vector_kernel_INT32_t_COMPLEX256_t(INT32_t m, INT32_t n, COMPLEX256_t *x, INT32_t incx, COMPLEX256_t *y, INT32_t incy,
         COMPLEX256_t *val, INT32_t *col, INT32_t *link, INT32_t *root):
    """
    Compute :math:`y = A^t * x`.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        col: C-contiguous C-array corresponding to vector ``A.col``.
        link: C-contiguous C-array corresponding to vector ``A.link``.
        root: C-contiguous C-array corresponding to vector ``A.root``.
    """
    cdef:
        COMPLEX256_t xi
        INT32_t i, k

    for i from 0 <= i < n:

        y[i*incy] = <COMPLEX256_t>(0.0+0.0j)


    for i from 0 <= i < m:
        xi = x[i*incx]
        k = root[i]

        while k != -1:
          y[col[k]*incy] += val[k] * xi
          k = link[k]
