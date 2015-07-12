########################################################################################################################
# Doesn't work!!!!
#
# Kept here for future reuse.
########################################################################################################################

"""
Several generic functions on types.
"""
from cysparse.types.cysparse_types cimport *
from cysparse.types.cysparse_types import *

cdef extern from "complex.h":
    float crealf(float complex z)
    float cimagf(float complex z)

    double creal(double complex z)
    double cimag(double complex z)

    long double creall(long double complex z)
    long double cimagl(long double complex z)

    double cabs(double complex z)
    float cabsf(float complex z)
    long double cabsl(long double complex z)

    double complex conj(double complex z)
    float complex  conjf (float complex z)
    long double complex conjl (long double complex z)

########################################################################################################################
# Split/join complex values
########################################################################################################################
# EXPLICIT TYPE TESTS


cdef split_array_complex_values_kernel_INT32_t_COMPLEX64_t(COMPLEX64_t * val,  val_length,
                                            FLOAT32_t * rval, INT32_t rval_length,
                                            FLOAT32_t * ival, INT32_t ival_length):

    if val_length > rval_length or val_length > ival_length:
        raise IndexError('Real and Imaginary values arrays must be of size equal or bigger as Complex array')

    cdef:
        INT32_t i
        COMPLEX64_t v

    for i from 0 <= i < val_length:
        v = val[i]
    
        rval[i] = crealf(v)
        ival[i] = cimagf(v)
    

cdef split_array_complex_values_kernel_INT32_t_COMPLEX128_t(COMPLEX128_t * val,  val_length,
                                            FLOAT64_t * rval, INT32_t rval_length,
                                            FLOAT64_t * ival, INT32_t ival_length):

    if val_length > rval_length or val_length > ival_length:
        raise IndexError('Real and Imaginary values arrays must be of size equal or bigger as Complex array')

    cdef:
        INT32_t i
        COMPLEX128_t v

    for i from 0 <= i < val_length:
        v = val[i]
    
        rval[i] = creal(v)
        ival[i] = cimag(v)
    

cdef split_array_complex_values_kernel_INT32_t_COMPLEX256_t(COMPLEX256_t * val,  val_length,
                                            FLOAT128_t * rval, INT32_t rval_length,
                                            FLOAT128_t * ival, INT32_t ival_length):

    if val_length > rval_length or val_length > ival_length:
        raise IndexError('Real and Imaginary values arrays must be of size equal or bigger as Complex array')

    cdef:
        INT32_t i
        COMPLEX256_t v

    for i from 0 <= i < val_length:
        v = val[i]
    
        rval[i] = creall(v)
        ival[i] = cimagl(v)
    



cdef split_array_complex_values_kernel_INT64_t_COMPLEX64_t(COMPLEX64_t * val,  val_length,
                                            FLOAT32_t * rval, INT64_t rval_length,
                                            FLOAT32_t * ival, INT64_t ival_length):

    if val_length > rval_length or val_length > ival_length:
        raise IndexError('Real and Imaginary values arrays must be of size equal or bigger as Complex array')

    cdef:
        INT64_t i
        COMPLEX64_t v

    for i from 0 <= i < val_length:
        v = val[i]
    
        rval[i] = crealf(v)
        ival[i] = cimagf(v)
    

cdef split_array_complex_values_kernel_INT64_t_COMPLEX128_t(COMPLEX128_t * val,  val_length,
                                            FLOAT64_t * rval, INT64_t rval_length,
                                            FLOAT64_t * ival, INT64_t ival_length):

    if val_length > rval_length or val_length > ival_length:
        raise IndexError('Real and Imaginary values arrays must be of size equal or bigger as Complex array')

    cdef:
        INT64_t i
        COMPLEX128_t v

    for i from 0 <= i < val_length:
        v = val[i]
    
        rval[i] = creal(v)
        ival[i] = cimag(v)
    

cdef split_array_complex_values_kernel_INT64_t_COMPLEX256_t(COMPLEX256_t * val,  val_length,
                                            FLOAT128_t * rval, INT64_t rval_length,
                                            FLOAT128_t * ival, INT64_t ival_length):

    if val_length > rval_length or val_length > ival_length:
        raise IndexError('Real and Imaginary values arrays must be of size equal or bigger as Complex array')

    cdef:
        INT64_t i
        COMPLEX256_t v

    for i from 0 <= i < val_length:
        v = val[i]
    
        rval[i] = creall(v)
        ival[i] = cimagl(v)
    





