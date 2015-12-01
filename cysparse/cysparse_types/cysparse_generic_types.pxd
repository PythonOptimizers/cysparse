########################################################################################################################
#
# Several helpers that deal with generic types.
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




cdef COMPLEX64_t make_complex_from_real_parts_COMPLEX64_t(FLOAT32_t real,
                                              FLOAT32_t imag)

cdef COMPLEX128_t make_complex_from_real_parts_COMPLEX128_t(FLOAT64_t real,
                                              FLOAT64_t imag)

cdef COMPLEX256_t make_complex_from_real_parts_COMPLEX256_t(FLOAT128_t real,
                                              FLOAT128_t imag)





cdef join_array_complex_values_kernel_INT32_t_COMPLEX64_t(
                                            FLOAT32_t * rval, INT32_t rval_length,
                                            FLOAT32_t * ival, INT32_t ival_length,
                                            COMPLEX64_t * val,  val_length)

cdef join_array_complex_values_kernel_INT32_t_COMPLEX128_t(
                                            FLOAT64_t * rval, INT32_t rval_length,
                                            FLOAT64_t * ival, INT32_t ival_length,
                                            COMPLEX128_t * val,  val_length)

cdef join_array_complex_values_kernel_INT32_t_COMPLEX256_t(
                                            FLOAT128_t * rval, INT32_t rval_length,
                                            FLOAT128_t * ival, INT32_t ival_length,
                                            COMPLEX256_t * val,  val_length)



cdef join_array_complex_values_kernel_INT64_t_COMPLEX64_t(
                                            FLOAT32_t * rval, INT64_t rval_length,
                                            FLOAT32_t * ival, INT64_t ival_length,
                                            COMPLEX64_t * val,  val_length)

cdef join_array_complex_values_kernel_INT64_t_COMPLEX128_t(
                                            FLOAT64_t * rval, INT64_t rval_length,
                                            FLOAT64_t * ival, INT64_t ival_length,
                                            COMPLEX128_t * val,  val_length)

cdef join_array_complex_values_kernel_INT64_t_COMPLEX256_t(
                                            FLOAT128_t * rval, INT64_t rval_length,
                                            FLOAT128_t * ival, INT64_t ival_length,
                                            COMPLEX256_t * val,  val_length)

