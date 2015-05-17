cdef update_ll_mat_matrix_from_c_arrays_indices_assign(LLSparseMatrix A, INT_t * index_i, Py_ssize_t index_i_length,
                                                       INT_t * index_j, Py_ssize_t index_j_length, object obj):
    """
    Update-assign (sub-)matrix: A[..., ...] = obj.

    Args:
        A: An :class:`LLSparseMatrix` object.
        index_i: C-arrays with ``INT_t`` indices.
        index_i_length: Length of ``index_i``.
        index_j: C-arrays with ``INT_t`` indices.
        index_j_length: Length of ``index_j``.
        obj: Any Python object that implements ``__getitem__()`` and accepts a ``tuple`` ``(i, j)``.

    Warning:
        There are not test whatsoever.
    """
    cdef:
        Py_ssize_t i
        Py_ssize_t j

    # TODO: use internal arrays like triplet (i, j, val)?
    # but indices can be anything...
    if PyLLSparseMatrix_Check(obj):
        #ll_mat =  obj
        for i from 0 <= i < index_i_length:
            for j from 0 <= j < index_j_length:
                A.put(index_i[i], index_j[j], obj[i, j])

    else:
        for i from 0 <= i < index_i_length:
            for j from 0 <= j < index_j_length:
                A.put(index_i[i], index_j[j], <FLOAT_t> obj[tuple(i, j)]) # not really optimized...
