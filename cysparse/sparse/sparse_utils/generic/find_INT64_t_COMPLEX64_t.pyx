from cysparse.types.cysparse_types cimport *

cdef extern from "math.h":
    double floor(double x)

cdef extern from "complex.h":
    float crealf(float complex z)
    float cimagf(float complex z)
    double creal(double complex z)
    double cimag(double complex z)
    long double creall(long double complex z)
    long double cimagl(long double complex z)


cdef INT32_t compare_complex_COMPLEX64_t(COMPLEX64_t a, COMPLEX64_t b):
    """
    Compare two complex numbers.

    If a < b, return -1, if a > b, return +1, if a == b, return 0.
    """

    if crealf(a) > crealf(b):
        return 1
    elif crealf(a) < crealf(b):
        return -1
    elif cimagf(a) > cimagf(b):
        # real parts are equal
        return 1
    elif cimagf(a) < cimagf(b):
        # real parts are equal
        return -1
    else:
        return 0



# EXPLICIT TYPE TESTS
cdef INT64_t find_bisec_INT64_t_COMPLEX64_t(COMPLEX64_t element, COMPLEX64_t * array, INT64_t lb, INT64_t ub)  except -1:
    """
    Find the index of a given element in an array by bissecting.

    Args:
        element: the element to find.
        array: the array to scour.
        lb: a lower bound on the search index.
        ub: an upper bound on the search index. Note that this bound is not reachable. Thus the search happens inside ``[lb, ub[``.

    Note:
        The array is supposed to be **sorted**. If the arrays contains the element multiple times, only the first one will be choosen,

    """
    assert lb < ub, "Lower bound must be smaller than upper bound"

    cdef:
        INT64_t lb_, ub_, middle_, index, i

    index = -1
    lb_ = lb
    ub_ = ub

    # EXPLICIT TYPE TESTS


    # complex number don't support comparison
    # but to find complex numbers in an array, this works perfectly
    while index == -1:
        middle_ = <INT64_t> floor((lb_ + ub_) / 2)
        if compare_complex_COMPLEX64_t(array[middle_], element) > 0:
            ub_ = middle_
        elif compare_complex_COMPLEX64_t(array[middle_], element) < 0:
            lb_ = middle_
        else:
            index = middle_

    # test if element found is the first one
    if index != -1:
        while index > lb and array[index] == element:
            index -= 1


    return index

cdef INT64_t find_linear_INT64_t_COMPLEX64_t(COMPLEX64_t element, COMPLEX64_t * array, INT64_t lb, INT64_t ub) except -1:
    """
    Find the index of a given element in an array by linear search.

    Args:
        element: the element to find.
        array: the array to scour.
        lb: a lower bound on the search index.
        ub: an upper bound on the search index. Note that this bound is not reachable. Thus the search happens inside ``[lb, ub[``.

    Returns:
        The index of the found element or the upper bound if the element is not found.

    Note:
        The array is **not** supposed to be **sorted**. If the arrays contains the element multiple times, only the first one will be choosen,

    """
    assert lb < ub, "Lower bound must be smaller than upper bound"

    cdef:
        INT64_t index

    for index from lb <= index < ub:
        if array[index] == element:
            break

    return index