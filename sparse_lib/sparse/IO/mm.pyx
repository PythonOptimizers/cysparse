from sparse_lib.cysparse_types cimport *

cdef:
    str MM_MATRIX_MARKET_BANNER_STR = "%%MatrixMarket"
    str MM_MTX_STR = "matrix"
    str MM_ARRAY_STR = "array"
    str MM_COORDINATE_STR = "coordinate"
    str MM_COMPLEX_STR = "complex"
    str MM_REAL_STR = "real"
    str MM_INT_STR = "integer"
    str MM_PATTERN_STR = "pattern"
    str MM_GENERAL_STR = "general"
    str MM_SYMM_STR = "symmetric"
    str MM_HERM_STR = "hermitian"
    str MM_SKEW_STR = "skew"

cdef enum:
    COMPLEX  = 0
    REAL     = 1
    INTEGER      = 2
    PATTERN  = 3

cdef enum:
    GENERAL   = 11
    HERMITIAN = 12
    SYMMETRIC = 13
    SKEW      = 14


include "mm_read_file.pxi"
include "mm_read_file2.pxi"