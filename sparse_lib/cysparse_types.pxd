# Adapt this file to your convenience
# 
# Types are given by XXX_t.
#
# For each type XXX_t, there is a corresponding:
#
# - XXX_T : an enum constant
# - XXX_t_BIT : numbers of bits for this type

#################################################################################################
#                                 *** COMPILATION CONSTANTS ***
#################################################################################################
# for huge matrices, we use unsigned long for size indices
DEF USE_HUGE_MATRIX = 0
# do we use double or float inside matrices?
DEF USE_DOUBLE_PRECISION = 1

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

cdef enum:
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

# we don't use neither the CPython nor the Python complex type
cdef struct complex64_t:
    FLOAT32_t real
    FLOAT32_t imag
    
cdef struct complex128_t:
    FLOAT64_t real
    FLOAT64_t imag

ctypedef complex64_t COMPLEX64_t                                    
ctypedef complex128_t COMPLEX128_t

#################################################################################################
#                                 *** BASIC RELATIVE TYPES ***
#
# SIZE_t: size of arrays, list, slices, matrices, ...
# INT_t: integers in general
# FLOAT_t: real fixed precision
# COMPLEX_t: complex fixed precision
#
# CHANGE THIS IF YOU REALLY KNOW WHAT YOU ARE DOING
#
#################################################################################################
IF USE_HUGE_MATRIX == 1:
    ctypedef UINT64_t SIZE_t
    ctypedef INT64_t INT_t

    cdef enum:
        SIZE_T = UINT64_T
        INT_T = INT64_T
ELSE:
    ctypedef UINT32_t SIZE_t
    ctypedef INT32_t INT_t
    
    cdef enum:
        SIZE_T = UINT32_T
        INT_T = INT32_T

IF USE_DOUBLE_PRECISION:
    ctypedef FLOAT64_t FLOAT_t
    ctypedef COMPLEX128_t COMPLEX_t
    
    cdef enum:
        FLOAT_T = FLOAT64_T
        COMPLEX_T = COMPLEX128_T
ELSE:
    ctypedef FLOAT32_t FLOAT_t
    ctypedef COMPLEX64_t COMPLEX_t

    cdef enum:
        FLOAT_T = FLOAT32_T
        COMPLEX_T = COMPLEX64_T



#################################################################################################
#                                 *** BASIC TYPES SIZES ***
#################################################################################################
cdef extern from "limits.h":
    enum:
        CHAR_BIT

# in bits
cdef enum:
    INT32_t_BIT = sizeof(INT32_t) * CHAR_BIT
    UINT32_t_BIT = sizeof(INT32_t) * CHAR_BIT
    INT64_t_BIT = sizeof(INT64_t) * CHAR_BIT
    UINT64_t_BIT = sizeof(UINT64_t) * CHAR_BIT
    FLOAT32_t_BIT = sizeof(FLOAT32_t) * CHAR_BIT
    FLOAT64_t_BIT = sizeof(FLOAT64_t) * CHAR_BIT
    COMPLEX64_t_BIT = sizeof(FLOAT32_t) * 2 * CHAR_BIT
    COMPLEX128_t_BIT = sizeof(FLOAT64_t) * 2 * CHAR_BIT
    
    INT_t_BIT = sizeof(INT_t) * CHAR_BIT
    FLOAT_t_BIT = sizeof(FLOAT_t) * CHAR_BIT
    SIZE_t_BIT = sizeof(SIZE_t) * CHAR_BIT
    COMPLEX_t_BIT = sizeof(FLOAT_t) * 2 * CHAR_BIT

