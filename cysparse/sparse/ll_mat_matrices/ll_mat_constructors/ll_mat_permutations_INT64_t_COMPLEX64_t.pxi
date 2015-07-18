

cdef LLSparseMatrix_INT64_t_COMPLEX64_t MakePermutationLLSparseMatrix_INT64_t_COMPLEX64_t(LLSparseMatrix_INT64_t_COMPLEX64_t A, cnp.ndarray numpy_vector):
    """
    Populate an ``LLSparseMatrix_INT64_t_COMPLEX64_t as a permutation matrix given a vector of permutation.

    See https://en.wikipedia.org/wiki/Permutation_matrix.

    Args:
        A: A ``LLSparseMatrix_INT64_t_COMPLEX64_t`` matrix.
        numpy_vector: A :program:`NumPy` array.

    Raise:
        ``MemoryError`` if not enough memory is available for internal computations.

    """
    cdef:
        INT64_t i, j, A_nrow, A_ncol

    A_nrow, A_ncol = A.shape

    if A_nrow != A_ncol:
        raise IndexError('nrow and ncol must be the same!')

    cdef INT64_t np_vec_size = <INT64_t> numpy_vector.size

    if np_vec_size < A_nrow:
        raise IndexError("Permutation vector must be at least as long as matrix's nrow")

    # test if numpy type can be cast to LLSparseMatrix types
    if not are_mixed_types_cast_compatible(A.itype, numpy_vector.dtype):
        raise TypeError('NumPy vector dtype (%s) not compatible with LLSparseMatrix itype (%s)' % (numpy_vector.dtype, type_to_string(A.itype)))

    for i from 0 <= i < np_vec_size:
        j = <COMPLEX64_t> numpy_vector[i]

        A.put(i, j, 1.0 + 0.0j)


    return A
