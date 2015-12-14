"""
Several routines to find elements in C-arrays.
"""

from cysparse.cysparse_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS

cdef INT32_t compare_complex_COMPLEX64_t(COMPLEX64_t a, COMPLEX64_t b)


cdef INT64_t find_bisec_INT64_t_COMPLEX64_t(COMPLEX64_t element, COMPLEX64_t * array, INT64_t lb, INT64_t ub) except -1
cdef INT64_t find_linear_INT64_t_COMPLEX64_t(COMPLEX64_t element, COMPLEX64_t * array, INT64_t lb, INT64_t ub) except -1