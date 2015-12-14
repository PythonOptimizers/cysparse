"""
Several routines to find elements in C-arrays.
"""

from cysparse.cysparse_types.cysparse_types cimport *

# EXPLICIT TYPE TESTS


cdef INT32_t find_bisec_INT32_t_FLOAT128_t(FLOAT128_t element, FLOAT128_t * array, INT32_t lb, INT32_t ub) except -1
cdef INT32_t find_linear_INT32_t_FLOAT128_t(FLOAT128_t element, FLOAT128_t * array, INT32_t lb, INT32_t ub) except -1