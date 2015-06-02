"""
This file contains the basic types used in CySparse.

CySparse allows you to dynamically type your matrices. This is not obvious to code in C or Cython. We do this by
generating different source code files for the different types. Sometimes, we need to test at compile time what type
we use (for instance to generate the right type of NumPy matrices). These tests are hard-coded.

You can find such places in the template files by looking for the comment '# EXPLICIT TYPE TESTS'.

Types are given by XXX_t.

For each type XXX_t, there is a corresponding:

 - XXX_T : an enum constant
 - XXX_t_BIT : numbers of bits for this type

Although we use NumPy compatible types, this will is not and should not be dependent on NumPy.

"""
from cpython cimport PyObject

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
# DO NOT CHANGE THIS (except if you **really** know what you are doing)
#
# We bring your attention to the following points:
#
# 1. A complex type that is at the same type fully compatible with NumPy, Python and Cython doesn't exist yet.
# 2. For the complex types, we use C99. These types are compatible with NumPy and we cast them for Python.
# 3. The code is specialized at some place. You can track those places by looking for the comment '# EXPLICIT TYPE TESTS'
#    If you want to remove or add a new type, you'll have to manually change the code at those places.
#    In short: DON'T CHANGE THE BASIC TYPES.
#
#################################################################################################

cpdef enum CySparseType:
    INT32_T = 0
    UINT32_T = 1
    INT64_T = 2
    UINT64_T = 3
    FLOAT32_T = 4
    FLOAT64_T = 5
    FLOAT128_T = 6
    COMPLEX64_T = 7
    COMPLEX128_T = 8
    COMPLEX256_T = 9

ctypedef int INT32_t
ctypedef unsigned int UINT32_t
ctypedef long INT64_t
ctypedef unsigned long UINT64_t

ctypedef float FLOAT32_t
ctypedef double FLOAT64_t
ctypedef long double FLOAT128_t

ctypedef float complex COMPLEX64_t
ctypedef double complex COMPLEX128_t
ctypedef long double complex COMPLEX256_t

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
    FLOAT128_t_BIT = sizeof(FLOAT128_t) * CHAR_BIT
    COMPLEX64_t_BIT = sizeof(COMPLEX64_t) * CHAR_BIT
    COMPLEX128_t_BIT = sizeof(COMPLEX128_t) * CHAR_BIT
    COMPLEX256_t_BIT = sizeof(COMPLEX256_t) * CHAR_BIT

#################################################################################################
#                                 *** SPARSE MATRIX TYPES ***
#################################################################################################
cdef struct CPType:
    CySparseType dtype
    CySparseType itype

#################################################################################################
#                                 *** BASIC TYPES LIMITS ***
#################################################################################################

# Warning: **don't** use the Cython version as they store the values inside an anonymous enum and thus cast everything
# before you even have a chance to grab the constants...
# TODO: is this system proof?
cdef extern from "stdint.h":
    cdef INT32_t INT32_MIN
    cdef INT32_t INT32_MAX
    cdef UINT32_t UINT32_MAX
    cdef INT64_t INT64_MIN
    cdef INT64_t INT64_MAX
    cdef UINT64_t UINT64_MAX

cdef extern from 'float.h':
    cdef FLOAT32_t FLT_MAX
    cdef FLOAT32_t FLT_MIN

    cdef FLOAT64_t DBL_MIN
    cdef FLOAT64_t DBL_MAX

    cdef FLOAT128_t LDBL_MIN
    cdef FLOAT128_t LDBL_MAX

########################################################################################################################
# Functions
########################################################################################################################
cpdef int result_type(CySparseType type1, CySparseType type2) except -1
cpdef int result_real_sum_type(CySparseType type1)

cpdef is_python_number(object obj)
cpdef is_cysparse_number(obj)
cpdef is_scalar(obj)

cpdef CySparseType min_integer_type(n, type_list) except? UINT64_T