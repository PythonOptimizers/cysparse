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

    Other cases are **not** needed as we can use 1.

3. :math:`A^h * b`: this **only** concerns complex matrices!

    - :program:`NumPy` array data C-contiguous, ``LLSparseMatrix`` not symmetric
    - :program:`NumPy` array data C-contiguous, ``LLSparseMatrix`` symmetric
    - :program:`NumPy` array data not C-contiguous, ``LLSparseMatrix`` not symmetric
    - :program:`NumPy` array data not C-contiguous, ``LLSparseMatrix`` symmetric [NOT DONE]

Note:
    We only consider C-arrays with same type of elements as the type of elements in the ``LLSparseMatrix``.
    Even if we construct the resulting :program:`NumPy` array as C-contiguous, the functions are more general and could
    be used with a given strided :program:`NumPy` `y` vector.
"""

########################################################################################################################
# A * b
########################################################################################################################

###########################################
# C-contiguous, non symmetric
###########################################
cdef void multiply_ll_mat_with_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, COMPLEX128_t *x, COMPLEX128_t *y,
         COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        COMPLEX128_t s
        INT64_t i, k

    for i from 0 <= i < m:

        s = <COMPLEX128_t>(0.0+0.0j)

        k = root[i]

        while k != -1:
          s += val[k] * x[col[k]]
          k = link[k]

        y[i] = s

###########################################
# C-contiguous, symmetric
###########################################
cdef void multiply_sym_ll_mat_with_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, COMPLEX128_t *x, COMPLEX128_t *y,
             COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        COMPLEX128_t s, v, xi
        INT64_t i, j, k

    for i from 0 <= i < m:
        xi = x[i]

        s = <COMPLEX128_t>(0.0+0.0j)

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
cdef void multiply_ll_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m,
            COMPLEX128_t *x, INT64_t incx,
            COMPLEX128_t *y, INT64_t incy,
            COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        COMPLEX128_t s
        INT64_t i, k

    for i from 0 <= i < m:

        s = <COMPLEX128_t>(0.0+0.0j)

        k = root[i]

        while k != -1:
            s += val[k] * x[col[k]*incx]
            k = link[k]

        y[i*incy] = s

###########################################
# Non C-contiguous, symmetric
###########################################
cdef void multiply_sym_ll_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m,
                COMPLEX128_t *x, INT64_t incx,
                COMPLEX128_t *y, INT64_t incy,
                COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        COMPLEX128_t s, v, xi
        INT64_t i, j, k

    for i from 0 <= i < m:
        xi = x[i*incx]

        s = <COMPLEX128_t>(0.0+0.0j)

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
# C-contiguous, non symmetric
###########################################
cdef void multiply_tranposed_ll_mat_with_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, INT64_t n, COMPLEX128_t *x, COMPLEX128_t *y,
         COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
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
        COMPLEX128_t xi
        INT64_t i, k

    for i from 0 <= i < n:

        y[i] = <COMPLEX128_t>(0.0+0.0j)


    for i from 0 <= i < m:
        xi = x[i]
        k = root[i]

        while k != -1:
          y[col[k]] += val[k] * xi
          k = link[k]

###########################################
# Non C-contiguous, non symmetric
###########################################
cdef void multiply_tranposed_ll_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, INT64_t n, COMPLEX128_t *x, INT64_t incx, COMPLEX128_t *y, INT64_t incy,
         COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
    """
    Compute :math:`y = A^t * x`.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as non C-contiguous (**with** strides).

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
        COMPLEX128_t xi
        INT64_t i, k

    for i from 0 <= i < n:

        y[i*incy] = <COMPLEX128_t>(0.0+0.0j)


    for i from 0 <= i < m:
        xi = x[i*incx]
        k = root[i]

        while k != -1:
          y[col[k]*incy] += val[k] * xi
          k = link[k]


########################################################################################################################
# A^h * b
########################################################################################################################

###########################################
# C-contiguous, non symmetric
###########################################
cdef void multiply_conjugate_tranposed_ll_mat_with_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, INT64_t n, COMPLEX128_t *x, COMPLEX128_t *y,
         COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
    """
    Compute :math:`y = A^h * x`.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).
        This version will **only** work for complex numbers and crashes at compile time for the other types.

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
        COMPLEX128_t xi
        INT64_t i, k

    for i from 0 <= i < n:
        y[i] = <COMPLEX128_t>(0.0+0.0j)

    for i from 0 <= i < m:
        xi = x[i]
        k = root[i]

        while k != -1:

            y[col[k]] += conj(val[k]) * xi


            k = link[k]


###########################################
# C-contiguous, symmetric
###########################################
cdef void multiply_conjugate_tranposed_sym_ll_mat_with_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, INT64_t n, COMPLEX128_t *x, COMPLEX128_t *y,
         COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
    """
    Compute :math:`y = A^h * x`.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).
        This version will **only** work for complex numbers and crashes at compile time for the other types.

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
        COMPLEX128_t xi, v
        INT64_t i, j, k

    for i from 0 <= i < n:
        y[i] = <COMPLEX128_t>(0.0+0.0j)

    for i from 0 <= i < m:
        xi = x[i]
        k = root[i]

        while k != -1:

            j = col[k]
            v = conj(val[k])
            y[j] += v * xi
            if i != j:
                y[i] += v * x[j]


            k = link[k]


###########################################
# Non C-contiguous, non symmetric
###########################################
cdef void multiply_conjugate_tranposed_ll_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, INT64_t n, COMPLEX128_t *x, INT64_t incx, COMPLEX128_t *y, INT64_t incy,
         COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
    """
    Compute :math:`y = A^h * x`.

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
        COMPLEX128_t xi
        INT64_t i, k

    for i from 0 <= i < n:
        y[i*incy] = <COMPLEX128_t>(0.0+0.0j)

    for i from 0 <= i < m:
        xi = x[i*incx]
        k = root[i]

        while k != -1:

            y[col[k]*incy] += conj(val[k]) * xi


            k = link[k]


###########################################
# Non C-contiguous, symmetric
###########################################
cdef void multiply_conjugate_tranposed_sym_ll_mat_with_strided_numpy_vector_kernel_INT64_t_COMPLEX128_t(INT64_t m, INT64_t n, COMPLEX128_t *x, INT64_t incx, COMPLEX128_t *y, INT64_t incy,
         COMPLEX128_t *val, INT64_t *col, INT64_t *link, INT64_t *root):
    """
    Compute :math:`y = A^h * x`.

    ``A`` is a symmetric :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as non C-contiguous (**with** strides).
        This version will **only** work for complex numbers and crashes at compile time for the other types.

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
        COMPLEX128_t xi, v
        INT64_t i, j, k

    for i from 0 <= i < n:
        y[i * incy] = <COMPLEX128_t>(0.0+0.0j)

    for i from 0 <= i < m:
        xi = x[i * incx]
        k = root[i]

        while k != -1:

            j = col[k]
            v = conj(val[k])
            y[j * incy] += v * xi
            if i != j:
                y[i * incy] += v * x[j * incx]


            k = link[k]

