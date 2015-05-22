"""
Several routines to multiply a :class:`LLSparseMatrix` matrix with a (comming from a :program:`NumPy` vector) C-array.

Covered cases:

- :program:`NumPy` array data C-contiguous, ``LLSparseMatrix`` no symmetric
- :program:`NumPy` array data C-contiguous, ``LLSparseMatrix`` symmetric
- :program:`NumPy` array data non C-contiguous, ``LLSparseMatrix`` non symmetric
- :program:`NumPy` array data non C-contiguous, ``LLSparseMatrix`` symmetric

Note:
    We only consider C-arrays with same type of elements as the type of elements in the ``LLSparseMatrix``.
"""

########################################################################################################################
# C-contiguous, no symmetric
########################################################################################################################
cdef void multiply_ll_mat_with_numpy_vector_kernel_INT64_t_FLOAT64_t(INT64_t m, FLOAT64_t *x, FLOAT64_t *y,
         FLOAT64_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        FLOAT64_t s
        INT64_t i, k

    for i from 0 <= i < m:

        s = <FLOAT64_t>0.0

        k = root[i]

        while k != -1:
          s += val[k] * x[col[k]]
          k = link[k]

        y[i] = s

########################################################################################################################
# C-contiguous, symmetric
########################################################################################################################
cdef void multiply_sym_ll_mat_with_numpy_vector_kernel_INT64_t_FLOAT64_t(INT64_t m, FLOAT64_t *x, FLOAT64_t *y,
             FLOAT64_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        FLOAT64_t s, v, xi
        INT64_t i, j, k

    for i from 0 <= i < m:
        xi = x[i]

        s = <FLOAT64_t>0.0

        k = root[i]

        while k != -1:
            j = col[k]
            v = val[k]
            s += v * x[j]
            if i != j:
                y[j] += v * xi
            k = link[k]

        y[i] = s

########################################################################################################################
# Non C-contiguous, non symmetric
########################################################################################################################
cdef void multiply_ll_mat_with_strided_numpy_vector_kernel_INT64_t_FLOAT64_t(INT64_t m,
            FLOAT64_t *x, INT64_t incx,
            FLOAT64_t *y, INT64_t incy,
            FLOAT64_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        FLOAT64_t s
        INT64_t i, k

    for i from 0 <= i < m:

        s = <FLOAT64_t>0.0

        k = root[i]

        while k != -1:
            s += val[k] * x[col[k]*incx]
            k = link[k]

        y[i*incy] = s

########################################################################################################################
# Non C-contiguous, symmetric
########################################################################################################################
cdef void multiply_sym_ll_mat_with_strided_numpy_vector_kernel_INT64_t_FLOAT64_t(INT64_t m,
                FLOAT64_t *x, INT64_t incx,
                FLOAT64_t *y, INT64_t incy,
                FLOAT64_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        FLOAT64_t s, v, xi
        INT64_t i, j, k

    for i from 0 <= i < m:
        xi = x[i*incx]

        s = <FLOAT64_t>0.0

        k = root[i]

        while k != -1:
            j = col[k]
            v = val[k]
            s += v * x[j*incx]
            if i != j:
                y[j*incy] += v * xi
            k = link[k]

        y[i*incy] = s