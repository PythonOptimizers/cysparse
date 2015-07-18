

cdef LLSparseMatrix_INT32_t_FLOAT32_t MakePermutationLLSparseMatrix_INT32_t_FLOAT32_t(LLSparseMatrix_INT32_t_FLOAT32_t A, cnp.ndarray numpy_vector):
    """
    Populate an ``LLSparseMatrix_INT32_t_FLOAT32_t as a permutation matrix given a vector of permutation.

    See https://en.wikipedia.org/wiki/Permutation_matrix.

    Args:
        A: A ``LLSparseMatrix_INT32_t_FLOAT32_t`` matrix.
        numpy_vector: A :program:`NumPy` array.

    Raise:
        ``MemoryError`` if not enough memory is available for internal computations.

    """
    cdef:
        INT32_t i, j, A_nrow, A_ncol

    A_nrow, A_ncol = A.shape

    if A_nrow != A_ncol:
        raise IndexError('nrow and ncol must be the same!')

    cdef INT32_t np_vec_size = <INT32_t> numpy_vector.size

    if np_vec_size < A_nrow:
        raise IndexError("Permutation vector must be at least as long as matrix's nrow")

    # test if numpy type can be cast to LLSparseMatrix types
    if not are_mixed_types_cast_compatible(A.itype, numpy_vector.dtype):
        raise TypeError('NumPy vector dtype (%s) not compatible with LLSparseMatrix itype (%s)' % (numpy_vector.dtype, type_to_string(A.itype)))

    for i from 0 <= i < np_vec_size:
        j = <FLOAT32_t> numpy_vector[i]

        A.put(i, j, <FLOAT32_t> 1.0)


    return A
