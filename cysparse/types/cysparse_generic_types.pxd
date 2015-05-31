from cysparse.types.cysparse_types cimport *


cdef test_cast_to_INT32_t(INT32_t n)

cdef test_cast_to_UINT32_t(UINT32_t n)

cdef test_cast_to_INT64_t(INT64_t n)

cdef test_cast_to_UINT64_t(UINT64_t n)

cdef test_cast_to_FLOAT32_t(FLOAT32_t n)

cdef test_cast_to_FLOAT64_t(FLOAT64_t n)

cdef test_cast_to_FLOAT128_t(FLOAT128_t n)

cdef test_cast_to_COMPLEX64_t(COMPLEX64_t n)

cdef test_cast_to_COMPLEX128_t(COMPLEX128_t n)

cdef test_cast_to_COMPLEX256_t(COMPLEX256_t n)


cdef min_type(n, type_list)