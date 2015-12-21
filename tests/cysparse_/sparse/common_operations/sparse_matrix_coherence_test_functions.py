"""
This module allows to test the coherence of all matrix-like objects.

As we can not force some inheritance (see documentation why), we have added some global functions that test
the presence of some methods inside a matrix-like class. This allows to test if **all** matrix-like objects share
a common interface. Ideally, this would be introduced by inheritance, directly or with mixins but this is currently
not possible.
"""


def common_matrix_like_attributes(obj):
    """
    Test the presence of some common parameters.

    Args:
        obj: The matrix-like object to test.

    Returns:
        ``True`` if all attributes do exist, ``False`` otherwise.
    """
    all_OK = True

    try:
        obj.nrow
        obj.ncol
        obj.nnz

    except AttributeError:
        all_OK = False

    return all_OK


def common_matrix_like_vector_multiplication(obj):
    """
    Test the presence of some common vector multiplication methods.

    Args:
        obj: The matrix-like object to test.

    Returns:
        ``True`` if all methods do exist, ``False`` otherwise.

    Note:
        The existence of a method doesn't guarantee anything about its implementation. Maybe it is not implemented (empty)
        or it raises some ``NotImplementedError`` or other exception.

    """
    all_OK = True

    try:
        getattr(obj, 'matvec')
        getattr(obj, 'matvec_transp')
        getattr(obj, 'matvec_htransp')
        getattr(obj, 'matvec_conj')

    except AttributeError:
        all_OK = False

    return all_OK


def common_matrix_like_matrix_multiplication(obj):
    """
    Test the presence of some common matrix multiplication methods.

    Args:
        obj: The matrix-like object to test.

    Returns:
        ``True`` if all methods do exist, ``False`` otherwise.

    Note:
        The existence of a method doesn't guarantee anything about its implementation. Maybe it is not implemented (empty)
        or it raises some ``NotImplementedError`` or other exception.

    """
    all_OK = True

    try:
        getattr(obj, 'matdot')
        getattr(obj, 'matdot_transp')

    except AttributeError:
        all_OK = False

    return all_OK