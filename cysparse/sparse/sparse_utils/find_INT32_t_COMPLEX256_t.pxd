"""
Several routines to find elements in C-arrays.
"""

from cysparse.types.cysparse_types cimport *

# EXPLICIT TYPE TESTS


cdef INT32_t find_bisec_INT32_t_COMPLEX256_t(COMPLEX256_t element, COMPLEX256_t * array, INT32_t lb, INT32_t ub) except -1
cdef INT32_t find_linear_INT32_t_COMPLEX256_t(COMPLEX256_t element, COMPLEX256_t * array, INT32_t lb, INT32_t ub) except -1