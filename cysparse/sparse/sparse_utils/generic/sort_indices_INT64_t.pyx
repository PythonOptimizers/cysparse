from cysparse.common_types.cysparse_types cimport *

cdef sort_array_INT64_t(INT64_t * a, INT64_t start, INT64_t end):
    """
    Sort array a between start and end - 1 (i.e. end **not** included).

    We use a basic insertion sort.
    """
    # TODO: put this is a new file and test
    cdef INT64_t i, j, value;

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