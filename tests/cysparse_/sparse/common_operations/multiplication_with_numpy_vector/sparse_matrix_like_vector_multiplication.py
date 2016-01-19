

def common_matrix_like_vector_multiplication(obj):
    """
    Test the presence of some common vector multiplication methods.

    Args:
        obj: The matrix-like object to test.

    Returns:
        A pair (``True``/``False``, ``problematic_operation``). ``True`` if all methods do exist, ``False`` otherwise. If
        ``False`` is returned, ``problematic_operation`` contains the name of the first problematic operation, i.e. the
        name of the first missing method.

    Note:
        The existence of a method doesn't guarantee anything about its implementation. Maybe it is not implemented (empty)
        or it raises some ``NotImplementedError`` or other exception.

    """
    all_OK = True
    problematic_operation = None

    try:
        problematic_operation = 'matvec'
        getattr(obj, 'matvec')
        problematic_operation = 'matvec_transp'
        getattr(obj, 'matvec_transp')
        problematic_operation = 'matvec_adj'
        getattr(obj, 'matvec_adj')
        problematic_operation = 'matvec_conj'
        getattr(obj, 'matvec_conj')

    except AttributeError:
        all_OK = False

    return all_OK, problematic_operation
