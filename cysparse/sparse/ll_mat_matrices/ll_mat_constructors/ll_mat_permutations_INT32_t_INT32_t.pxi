

cdef LLSparseMatrix_INT32_t_INT32_t MakePermutationLLSparseMatrix_INT32_t_INT32_t(LLSparseMatrix_INT32_t_INT32_t A, cnp.ndarray numpy_vector):
    """
    Populate an ``LLSparseMatrix_INT32_t_INT32_t as a permutation matrix given a vector of permutation.

    See https://en.wikipedia.org/wiki/Permutation_matrix.

    Args:
        A: A ``LLSparseMatrix_INT32_t_INT32_t`` matrix.
        numpy_vector: A :program:`NumPy` array.

    Raise:
        ``MemoryError`` if not enough memory is available for internal computations.
        ``IndexError`` if the :program:`NumPy` vector is not long enough or if the :class:`LLSparseMatrix` is not a
        square matrix.
        ``TypeError`` if the :program:`NumPy` vector dtype is not compatible with the :class:`LLSparseMatrix` itype.
        Whenever ``test`` is set to ``True``, an internal array is constructed and if there is not enough memory, a ``MemoryError``
        is raised. Other errors can be raised if the content of the :program:`NumPy` vector doesn't correspond to matrix indices.

    Warning:
        The :program:`NumPy` vector must be compatible with the :class:`LLSparseMatrix` ``itype``.

    Note:
        A test is done to assert that the vector is indeed a permutation vector.

    """
    cdef:
        INT32_t i, j, A_nrow, A_ncol
        INT32_t * one_index_columns # for testing

    A_nrow, A_ncol = A.shape

    if A_nrow != A_ncol:
        raise IndexError('nrow and ncol must be the same!')

    cdef INT32_t np_vec_size = <INT32_t> numpy_vector.size

    if np_vec_size < A_nrow:
        raise IndexError("Permutation vector must be at least as long as matrix's nrow")

    # test if numpy type can be cast to LLSparseMatrix types
    if not are_mixed_types_cast_compatible(A.itype, numpy_vector.dtype):
        raise TypeError('NumPy vector dtype (%s) not compatible with LLSparseMatrix itype (%s)' % (numpy_vector.dtype, type_to_string(A.itype)))

    one_index_columns = <INT32_t *> PyMem_Malloc(A_nrow * sizeof(INT32_t))
    if not one_index_columns:
        raise MemoryError()

    for i from 0 <= i < A_nrow:
        one_index_columns[i] = 0

    try:

        for i from 0 <= i < A_nrow:
            j = <INT32_t> numpy_vector[i]

            if one_index_columns[j] == 0:
                one_index_columns[j] = 1
            else:
                raise IndexError('NumPy vector contains an index %d twice at %d' % (j, i))



            A.put(i, j, <INT32_t> 1.0)

    except:
        PyMem_Free(one_index_columns)
        raise

    return A