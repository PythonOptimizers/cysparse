"""
Matrix - vector multiplication kernels for REAL NUMBERS
"""
# C-contiguous, no symmetric
cdef void multiply_ll_mat_with_numpy_vector_kernel(INT_t m, FLOAT_t *x, FLOAT_t *y,
         FLOAT_t *val, INT_t *col, INT_t *link, INT_t *root):
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
        FLOAT_t s
        INT_t i, k

    for i from 0 <= i < m:
        s = 0.0
        k = root[i]

        while k != -1:
          s += val[k] * x[col[k]]
          k = link[k]

        y[i] = s

# C-contiguous, symmetric
cdef void multiply_sym_ll_mat_with_numpy_vector_kernel(INT_t m, FLOAT_t *x, FLOAT_t *y,
             FLOAT_t *val, INT_t *col, INT_t *link, INT_t *root):
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
        FLOAT_t s, v, xi
        INT_t i, j, k

    for i from 0 <= i < m:
        xi = x[i]
        s = 0.0
        k = root[i]

        while k != -1:
            j = col[k]
            v = val[k]
            s += v * x[j]
            if i != j:
                y[j] += v * xi
            k = link[k]

        y[i] = s

# Non C-contiguous, non symmetric
cdef void multiply_ll_mat_with_strided_numpy_vector_kernel(INT_t m,
            FLOAT_t *x, INT_t incx,
            FLOAT_t *y, INT_t incy,
            FLOAT_t *val, INT_t *col, INT_t *link, INT_t *root):
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
        FLOAT_t s
        INT_t i, k

    for i from 0 <= i < m:
        s = 0.0
        k = root[i]

        while k != -1:
            s += val[k] * x[col[k]*incx]
            k = link[k]

        y[i*incy] = s

# Non C-contiguous, non symmetric
cdef void multiply_sym_ll_mat_with_strided_numpy_vector_kernel(INT_t m,
                FLOAT_t *x, INT_t incx,
                FLOAT_t *y, INT_t incy,
                FLOAT_t *val, INT_t *col, INT_t *link, INT_t *root):
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
        FLOAT_t s, v, xi
        INT_t i, j, k

    for i from 0 <= i < m:
        xi = x[i*incx]
        s = 0.0
        k = root[i]

        while k != -1:
            j = col[k]
            v = val[k]
            s += v * x[j*incx]
            if i != j:
                y[j*incy] += v * xi
            k = link[k]

        y[i*incy] = s
