#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False
    
from cysparse.common_types.cysparse_types cimport *


cdef element_to_string_COMPLEX128_t(COMPLEX128_t v, int cell_width=?)
cdef conjugated_element_to_string_COMPLEX128_t(COMPLEX128_t v, int cell_width=?)
cdef empty_to_string_COMPLEX128_t(int cell_width=?)