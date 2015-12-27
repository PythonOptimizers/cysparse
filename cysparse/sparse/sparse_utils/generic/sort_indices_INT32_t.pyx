from cysparse.common_types.cysparse_types cimport *

cdef sort_array_INT32_t(INT32_t * a, INT32_t start, INT32_t end):
    """
    Sort array a between start and end - 1 (i.e. end **not** included).

    We use a basic insertion sort.
    """
    # TODO: put this is a new file and test
    cdef INT32_t i, j, value;

    i = start

    while i < end:

        value = a[i]
        j = i - 1
        while j >= start and a[j] > value:
            # shift
            a[j+1] = a[j]

            j -= 1

        # place key at right place
        a[j+1] = value

        i += 1