# Adapt this file to your convenience
# 
# Types are given by XXX_t.
#
# For each type XXX_t, there is a corresponding:
#
# - XXX_T : an enum constant
# - XXX_t_BIT : numbers of bits for this type

cimport numpy as cnp

#################################################################################################
#                                 *** BASIC ABSOLUTE TYPES ***
#
# INT32_t
# UINT32_t
# INT64_t
# UINT64_t
# FLOAT32_t
# FLOAT64_t
# COMPLEX64_t
# COMPLEX128_t
#
# DO NOT CHANGE THIS
#################################################################################################

cpdef enum CySparseType:
    INT32_T = 0
    UINT32_T = 1
    INT64_T = 2
    UINT64_T = 3
    FLOAT32_T = 4
    FLOAT64_T = 5
    COMPLEX64_T = 6
    COMPLEX128_T = 7

ctypedef int INT32_t
ctypedef unsigned int UINT32_t
ctypedef long INT64_t
ctypedef unsigned long UINT64_t

ctypedef float FLOAT32_t
ctypedef double FLOAT64_t

#ctypedef cnp.complex64_t COMPLEX64_t
ctypedef float complex COMPLEX64_t
#ctypedef cnp.complex128_t COMPLEX128_t
ctypedef double complex COMPLEX128_t

# elements that behave like integers
INTEGER_ELEMENT_TYPES = [INT32_T, UINT32_T, INT64_T, UINT64_T]
# elements that behave like floats
FLOAT_ELEMENT_TYPES = [FLOAT32_T, FLOAT64_T]
# elements that behave like complex numbers (we only consider complex floats)
COMPLEX_ELEMENT_TYPES = [COMPLEX64_T, COMPLEX128_T]

#################################################################################################
#                                 *** BASIC TYPES SIZES ***
#################################################################################################
cdef extern from "limits.h":
    enum:
        CHAR_BIT

# in bits
cdef enum CySparseTypeBitSize:
    INT32_t_BIT = sizeof(INT32_t) * CHAR_BIT
    UINT32_t_BIT = sizeof(INT32_t) * CHAR_BIT
    INT64_t_BIT = sizeof(INT64_t) * CHAR_BIT
    UINT64_t_BIT = sizeof(UINT64_t) * CHAR_BIT
    FLOAT32_t_BIT = sizeof(FLOAT32_t) * CHAR_BIT
    FLOAT64_t_BIT = sizeof(FLOAT64_t) * CHAR_BIT
    COMPLEX64_t_BIT = sizeof(COMPLEX64_t) * CHAR_BIT
    COMPLEX128_t_BIT = sizeof(COMPLEX128_t) * CHAR_BIT

#################################################################################################
#                                 *** SPARSE MATRIX TYPES ***
#################################################################################################
cdef struct CPType:
    CySparseType dtype
    CySparseType itype