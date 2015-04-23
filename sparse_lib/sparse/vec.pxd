
cdef class DVector:
    cdef double *data
    cdef public int n


cdef class IVector:
    cdef int* data
    cdef public int n



cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size