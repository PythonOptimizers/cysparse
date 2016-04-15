"""
Several routines to multiply a :class:`CSCSparseMatrix` matrix with a (comming from a :program:`NumPy` vector) C-array.

Covered cases:

1. :math:`A * b`:

- :program:`NumPy` array data C-contiguous, ``CSCSparseMatrix`` not symmetric
- :program:`NumPy` array data C-contiguous, ``CSCSparseMatrix`` symmetric
- :program:`NumPy` array data not C-contiguous, ``CSCSparseMatrix`` not symmetric
- :program:`NumPy` array data not C-contiguous, ``CSCSparseMatrix`` symmetric

2. :math:`A^t * b`

- :program:`NumPy` array data C-contiguous, ``CSCSparseMatrix`` not symmetric
- :program:`NumPy` array data not C-contiguous, ``CSCSparseMatrix`` not symmetric

3. :math:`A^h * b`

- :program:`NumPy` array data C-contiguous, ``CSCSparseMatrix`` not symmetric
- :program:`NumPy` array data C-contiguous, ``CSCSparseMatrix`` symmetric
- :program:`NumPy` array data not C-contiguous, ``CSCSparseMatrix`` not symmetric
- :program:`NumPy` array data not C-contiguous, ``CSCSparseMatrix`` symmetric

4. :math:`\textrm{conj}(A) * b`

- :program:`NumPy` array data C-contiguous, ``CSCSparseMatrix`` not symmetric
- :program:`NumPy` array data C-contiguous, ``CSCSparseMatrix`` symmetric
- :program:`NumPy` array data not C-contiguous, ``CSCSparseMatrix`` not symmetric
- :program:`NumPy` array data not C-contiguous, ``CSCSparseMatrix`` symmetric

Note:
    We only consider C-arrays with same type of elements as the type of elements in the ``CSCSparseMatrix``.
    Even if we construct the resulting :program:`NumPy` array as C-contiguous, the functions are more general and could
    be used with a given strided :program:`NumPy` `y` vector.
"""

#from libc.string cimport memset # TODO: get rid of this...



########################################################################################################################
# A * b
########################################################################################################################

###########################################
# C-contiguous, non symmetric
###########################################
cdef void multiply_csc_mat_with_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, COMPLEX64_t *y,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind, bint init_y = 1):
    """
    Compute ``y = A * x``.

    ``A`` is a :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t i, j, k

    #if init_y:
    #    memset(y, 0, m*sizeof(y))

    if init_y:
        # init numpy array
        for i from 0 <= i < m:

            y[i] = <COMPLEX64_t>(0.0+0.0j)



    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:
            i = row[k]
            y[i] += val[k] * x[j]


###########################################
# C-contiguous, symmetric
###########################################
cdef void multiply_sym_csc_mat_with_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, COMPLEX64_t *y,
             COMPLEX64_t *val, INT32_t *row, INT32_t *ind, bint init_y = 1):
    """
    Compute ``y = A * x``.

    ``A`` is a **symmetric** :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t i, j, k

    #if init_y:
    #    memset(y, 0, m*sizeof(y))

    if init_y:
        # init numpy array
        for i from 0 <= i < m:

            y[i] = <COMPLEX64_t>(0.0+0.0j)


    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:
            i = row[k]
            y[i] += val[k] * x[j]
            if j != i:
                y[j] += val[k] * x[i]


###########################################
# Non C-contiguous, non symmetric
###########################################
cdef void multiply_csc_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n,
            COMPLEX64_t *x, INT32_t incx,
            COMPLEX64_t *y, INT32_t incy,
            COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute ``y = A * x``.

    ``A`` is :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional **non** C-contiguous numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider *both* numpy arrays as **non** C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t i, j, k


    # init numpy array
    for i from 0 <= i < m:

        y[i * incy] = <COMPLEX64_t>(0.0+0.0j)


    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:
            i = row[k]
            y[i * incy] += val[k] * x[j * incx]


###########################################
# Non C-contiguous, symmetric
###########################################
cdef void multiply_sym_csc_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n,
                COMPLEX64_t *x, INT32_t incx,
                COMPLEX64_t *y, INT32_t incy,
                COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute ``y = A * x``.

    ``A`` is a **symmetric** :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional **non** C-contiguous numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider *both* numpy arrays as **non** C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t i, j, k

    # init numpy array
    for i from 0 <= i < m:

        y[i * incy] = <COMPLEX64_t>(0.0+0.0j)


    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:
            i = row[k]
            y[i * incy] += val[k] * x[j * incx]
            if j != i:
                y[j * incy] += val[k] * x[i * incx]



########################################################################################################################
# A^t * b
########################################################################################################################

###########################################
# C-contiguous, non symmetric
###########################################
cdef void multiply_tranposed_csc_mat_with_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, COMPLEX64_t *y,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute :math:`y = A^t * x`.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        COMPLEX64_t s
        INT32_t j, k

    for j from 0 <= j < n:

        s = <COMPLEX64_t>(0.0+0.0j)

        for k from ind[j]<= k < ind[j+1]:
            s += val[k] * x[row[k]]

        y[j] = s


###########################################
# Non C-contiguous, non symmetric
###########################################
cdef void multiply_tranposed_csc_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, INT32_t incx, COMPLEX64_t *y, INT32_t incy,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute :math:`y = A^t * x`.

    ``A`` is a :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as non C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        COMPLEX64_t s
        INT32_t j, k

    for j from 0 <= j < n:

        s = <COMPLEX64_t>(0.0+0.0j)

        for k from ind[j]<= k < ind[j+1]:
            s += val[k] * x[row[k] * incx]

        y[j * incy] = s


########################################################################################################################
# A^h * b
########################################################################################################################

###########################################
# C-contiguous, non symmetric
###########################################
cdef void multiply_conjugate_transposed_csc_mat_with_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, COMPLEX64_t *y,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute :math:`y = A^h * x`.

    ``A`` is a :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        COMPLEX64_t s
        INT32_t j, k

    for j from 0 <= j < n:
        s = <COMPLEX64_t>(0.0+0.0j)

        for k from ind[j]<= k < ind[j+1]:

            s += conjf(val[k]) * x[row[k]]


        y[j] = s

###########################################
# C-contiguous, symmetric
###########################################
cdef void multiply_conjugate_transposed_sym_csc_mat_with_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, COMPLEX64_t *y,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute :math:`y = A^h * x`.

    ``A`` is a **symmetric** :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        COMPLEX64_t v
        INT32_t i, j, k

    # init numpy array
    for j from 0 <= j < n:
        y[j] = <COMPLEX64_t>(0.0+0.0j)

    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:
            i = row[k]

            v = conjf(val[k])


            y[j] += v * x[i]

            if i != j:
                y[i] += v * x[j]


###########################################
# Non C-contiguous, non symmetric
###########################################
cdef void multiply_conjugate_tranposed_csc_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, INT32_t incx, COMPLEX64_t *y, INT32_t incy,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute :math:`y = A^h * x`.

    ``A`` is a :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as non C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        COMPLEX64_t s
        INT32_t j, k

    for j from 0 <= j < n:
        s = <COMPLEX64_t>(0.0+0.0j)

        for k from ind[j]<= k < ind[j+1]:

            s += conjf(val[k]) * x[row[k] * incx]


        y[j * incy] = s


###########################################
# Non C-contiguous, symmetric
###########################################
cdef void multiply_conjugate_transposed_sym_csc_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n,
    COMPLEX64_t *x, INT32_t incx,COMPLEX64_t *y, INT32_t incy,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute :math:`y = A^h * x`.

    ``A`` is a **symmetric** :class:`LLSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as non C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        COMPLEX64_t v
        INT32_t i, j, k

    # init numpy array
    for j from 0 <= j < n:
        y[j * incy] = <COMPLEX64_t>(0.0+0.0j)

    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:
            i = row[k]

            v = conjf(val[k])


            y[j * incy] += v * x[i * incx]

            if i != j:
                y[i * incy] += v * x[j * incx]


########################################################################################################################
# conj(A) * b
########################################################################################################################

###########################################
# C-contiguous, non symmetric
###########################################
cdef void multiply_conjugated_csc_mat_with_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, COMPLEX64_t *y,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute ``y = conj(A) * x``.

    ``A`` is a :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t i, j, k
        COMPLEX64_t v

    # init numpy array
    for i from 0 <= i < m:
        y[i] = <COMPLEX64_t>(0.0+0.0j)

    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:

            v = conjf(val[k])

            i = row[k]
            y[i] += v * x[j]

###########################################
# C-contiguous, symmetric
###########################################
cdef void multiply_conjugated_sym_csc_mat_with_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n, COMPLEX64_t *x, COMPLEX64_t *y,
             COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute ``y = conj(A) * x``.

    ``A`` is a **symmetric** :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as C-contiguous (**without** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: C-contiguous C-array corresponding to vector ``x``.
        y: C-contiguous C-array corresponding to vector ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t i, j, k
        COMPLEX64_t v

    # init numpy array
    for i from 0 <= i < m:
        y[i] = <COMPLEX64_t>(0.0+0.0j)

    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:

            v = conjf(val[k])

            i = row[k]

            y[i] += v * x[j]
            if j != i:
                y[j] += v * x[i]

###########################################
# Non C-contiguous, non symmetric
###########################################
cdef void multiply_conjugated_csc_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n,
         COMPLEX64_t *x, INT32_t incx,
         COMPLEX64_t *y, INT32_t incy,
         COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute ``y = conj(A) * x``.

    ``A`` is a :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as non C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: (non necessarily C-contiguous) C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: (non necessarily C-contiguous) C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t i, j, k
        COMPLEX64_t v

    # init numpy array
    for i from 0 <= i < m:
        y[i * incy] = <COMPLEX64_t>(0.0+0.0j)

    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:

            v = conjf(val[k])

            i = row[k]
            y[i * incy] += v * x[j * incx]

###########################################
# Non C-contiguous, symmetric
###########################################
cdef void multiply_conjugated_sym_csc_mat_with_strided_numpy_vector_kernel_INT32_t_COMPLEX64_t(INT32_t m, INT32_t n,
            COMPLEX64_t *x, INT32_t incx, COMPLEX64_t *y, INT32_t incy,
            COMPLEX64_t *val, INT32_t *row, INT32_t *ind):
    """
    Compute ``y = conj(A) * x``.

    ``A`` is a **symmetric** :class:`CSCSparseMatrix` and ``x`` and ``y`` are one dimensional numpy arrays.
    In this kernel function, we only use the corresponding C-arrays.

    Warning:
        This version consider the arrays as non C-contiguous (**with** strides).

    Args:
        m: Number of rows of the matrix ``A``.
        n: Number of columns of the matrix ``A``.
        x: (non necessarily C-contiguous) C-array corresponding to vector ``x``.
        incx: Stride for array ``x``.
        y: (non necessarily C-contiguous) C-array corresponding to vector ``y``.
        incy: Stride for array ``y``.
        val: C-contiguous C-array corresponding to vector ``A.val``.
        row: C-contiguous C-array corresponding to vector ``A.row``.
        ind: C-contiguous C-array corresponding to vector ``A.ind``.
    """
    cdef:
        INT32_t i, j, k
        COMPLEX64_t v

    # init numpy array
    for i from 0 <= i < m:

        y[i * incy] = <COMPLEX64_t>(0.0+0.0j)


    # multiplication, column-wise...
    for j from 0 <= j < n:
        for k from ind[j]<= k < ind[j+1]:

            v = conjf(val[k])

            i = row[k]

            y[i * incy] += v * x[j * incx]
            if j != i:
                y[j * incy] += v * x[i * incx]


