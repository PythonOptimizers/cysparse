from cysparse.types.cysparse_types cimport *

cdef extern from "math.h":
    double floor(double x)

cdef extern from "complex.h":
    float crealf(float complex z)
    float cimagf(float complex z)
    double creal(double complex z)
    double cimag(double complex z)



# EXPLICIT TYPE TESTS
cdef INT64_t find_bisec_INT64_t_INT32_t(INT32_t element, INT32_t * array, INT64_t lb, INT64_t ub)  except -1:
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


    while index == -1:
        middle_ = <INT64_t> floor((lb_ + ub_) / 2)

        if array[middle_] > element:
            ub_ = middle_
        elif array[middle_] < element:
            lb_ = middle_
        else:
            index = middle_

    # test if element found is the first one
    if index != -1:
        while index > lb and array[index] == element:
            index -= 1


    return index

cdef INT64_t find_linear_INT64_t_INT32_t(INT32_t element, INT32_t * array, INT64_t lb, INT64_t ub) except -1:
    """
    Find the index of a given element in an array by linear search.

    Args:
        element: the element to find.
        array: the array to scour.
        lb: a lower bound on the search index.
        ub: an upper bound on the search index. Note that this bound is not reachable. Thus the search happens inside ``[lb, ub[``.

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