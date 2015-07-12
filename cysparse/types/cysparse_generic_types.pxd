########################################################################################################################
# Several helpers that deal with generic types.
#
#
########################################################################################################################
from cysparse.types.cysparse_types cimport *



cdef split_array_complex_values_kernel_INT32_t_COMPLEX64_t(COMPLEX64_t * val,  val_length,
                                            FLOAT32_t * rval, INT32_t rval_length,
                                            FLOAT32_t * ival, INT32_t ival_length)

cdef split_array_complex_values_kernel_INT32_t_COMPLEX128_t(COMPLEX128_t * val,  val_length,
                                            FLOAT64_t * rval, INT32_t rval_length,
                                            FLOAT64_t * ival, INT32_t ival_length)

cdef split_array_complex_values_kernel_INT32_t_COMPLEX256_t(COMPLEX256_t * val,  val_length,
                                            FLOAT128_t * rval, INT32_t rval_length,
                                            FLOAT128_t * ival, INT32_t ival_length)



cdef split_array_complex_values_kernel_INT64_t_COMPLEX64_t(COMPLEX64_t * val,  val_length,
                                            FLOAT32_t * rval, INT64_t rval_length,
                                            FLOAT32_t * ival, INT64_t ival_length)

cdef split_array_complex_values_kernel_INT64_t_COMPLEX128_t(COMPLEX128_t * val,  val_length,
                                            FLOAT64_t * rval, INT64_t rval_length,
                                            FLOAT64_t * ival, INT64_t ival_length)

cdef split_array_complex_values_kernel_INT64_t_COMPLEX256_t(COMPLEX256_t * val,  val_length,
                                            FLOAT128_t * rval, INT64_t rval_length,
                                            FLOAT128_t * ival, INT64_t ival_length)

