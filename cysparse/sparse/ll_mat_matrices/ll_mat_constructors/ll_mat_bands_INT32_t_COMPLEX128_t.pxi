

cdef LLSparseMatrix_INT32_t_COMPLEX128_t MakeBandLLSparseMatrix_INT32_t_COMPLEX128_t(LLSparseMatrix_INT32_t_COMPLEX128_t A, diag_coeff, list numpy_arrays):
    """
    Populate an ``LLSparseMatrix_INT32_t_COMPLEX128_t with diagonals.

    See https://en.wikipedia.org/wiki/Band_matrix.

    Args:
        A: A ``LLSparseMatrix_INT32_t_COMPLEX128_t`` matrix.
        diag_coeff: A list or slice with diagonal indices.
        numpy_arrays: A list of :program:`NumPy` arrays.

    Raise:
        ``IndexError`` if number of **accepted** diagonal numbers doesn't match the number of :program:`NumPy` arrays given as arguements.
        ``RuntimeError`` if slice could not be translated.
        ``MemoryError`` if not enough memory is available for internal computations.

    Note:
        We don't expect the matrix to be square. The added diagonals can be anywhere in the matrix, not only together as a band.
    """
    cdef:
        INT32_t i, j, A_nrow, A_ncol
        INT32_t ret
        Py_ssize_t start, stop, step, length, index, max_length
        INT32_t * indices
        PyObject *val

    A_nrow, A_ncol = A.shape

    cdef PyObject * obj = <PyObject *> diag_coeff

    # normally, with slices, it is common in Python to chop off...
    # Here we only chop off from above, not below...
    # -m + 1 <= k <= n -1   : only k <= n - 1 will be satified (greater indices are disregarded)
    # but nothing is done if k < -m + 1
    max_length = A.__ncol

    # TODO: put this in its own method?
    # grab diag coefficients
    if PySlice_Check(obj):
        # slice
        ret = PySlice_GetIndicesEx(<PySliceObject*>obj, max_length, &start, &stop, &step, &length)
        if ret:
            raise RuntimeError("Slice could not be translated")

        #print "start, stop, step, length = (%d, %d, %d, %d)" % (start, stop, step, length)

        indices = <INT32_t *> PyMem_Malloc(length * sizeof(INT32_t))
        if not indices:
            raise MemoryError()

        # populate indices
        i = start
        for j from 0 <= j < length:
            indices[j] = i
            i += step

    elif PyList_Check(obj):
        length = PyList_Size(obj)
        indices = <INT32_t *> PyMem_Malloc(length * sizeof(INT32_t))
        if not indices:
            raise MemoryError()

        for i from 0 <= i < length:
            val = PyList_GetItem(obj, <Py_ssize_t>i)
            if PyInt_Check(val):
                index = PyInt_AS_LONG(val)
                indices[i] = <INT32_t> index
            else:
                PyMem_Free(indices)
                raise ValueError("List must only contain integers")
    else:
        raise TypeError("Index object is not recognized (list or slice)")

    # test lengths of inputs
    if length != len(numpy_arrays):
        raise IndexError("Number of accepted diagonal numbers (%d) doesn't match number of NumPy arrays (%d)" % (length, len(numpy_arrays)))

    for i from 0 <= i < length:
        A.put_diagonal(indices[i], numpy_arrays[i])

    return A
