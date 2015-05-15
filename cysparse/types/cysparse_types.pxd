# Adapt this file to your convenience
# 
# Types are given by XXX_t.
#
# For each type XXX_t, there is a corresponding:
#
# - XXX_T : an enum constant
# - XXX_t_BIT : numbers of bits for this type


#################################################################################################
#                                 *** BASIC ABSOLUTE TYPES ***
#
# INT32_t
# UINT32_t
# INT64_t
# UINT64_t
# FLOAT32_t
# FLOAT64_t
#
# DO NOT CHANGE THIS
#################################################################################################

cdef enum CySparseType:
    INT32_T = 0
    UINT32_T = 1
    INT64_T = 2
    UINT64_T = 3
    FLOAT32_T = 4
    FLOAT64_T = 5

ctypedef int INT32_t
ctypedef unsigned int UINT32_t
ctypedef long INT64_t
ctypedef unsigned long UINT64_t

ctypedef float FLOAT32_t
ctypedef double FLOAT64_t


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

#################################################################################################
#                                 *** SPARSE MATRIX TYPES ***
#################################################################################################
cdef struct CPType:
    CySparseType dtype
    CySparseType itype