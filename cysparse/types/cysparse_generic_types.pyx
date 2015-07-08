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

########################################################################################################################
# Tests on numbers
########################################################################################################################
# EXPLICIT TYPE TESTS

cdef test_cast_to_INT32_t(INT32_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (INT32_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef INT32_t n_ = n

cdef test_cast_to_UINT32_t(UINT32_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (UINT32_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef UINT32_t n_ = n

cdef test_cast_to_INT64_t(INT64_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (INT64_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef INT64_t n_ = n

cdef test_cast_to_UINT64_t(UINT64_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (UINT64_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef UINT64_t n_ = n

cdef test_cast_to_FLOAT32_t(FLOAT32_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (FLOAT32_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef FLOAT32_t n_ = n

cdef test_cast_to_FLOAT64_t(FLOAT64_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (FLOAT64_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef FLOAT64_t n_ = n

cdef test_cast_to_FLOAT128_t(FLOAT128_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (FLOAT128_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef FLOAT128_t n_ = n

cdef test_cast_to_COMPLEX64_t(COMPLEX64_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (COMPLEX64_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef COMPLEX64_t n_ = n

cdef test_cast_to_COMPLEX128_t(COMPLEX128_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (COMPLEX128_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef COMPLEX128_t n_ = n

cdef test_cast_to_COMPLEX256_t(COMPLEX256_t n):
    """
    Little function to test if casting is possible or not thanks to Cython overflow check.

    In itself, this function doesn't do anything but if the arguement is not accepted, Cython throws at runtime an ``OverflowError``.

    Args:
        n (COMPLEX256_t): Number to cast.

    Raises:
        ``OverflowError`` is casting is **not** possible.

    Note:
        This is a meta function. Don't use it unless you really know what you are doing.
    """
    cdef COMPLEX256_t n_ = n


# EXPLICIT TYPE TESTS
cdef min_type(n, type_list):
    """
    Return the minimal type that can `n` can be casted into from a list of types.

    Note:
        We suppose that the list is sorted by ascending types.

    Raises:
        ``TypeError`` if no type can be used to cast `n` or if one element in the type list is not recognized as a
        ``CySparseType``.

    Args:
        n: Python number to cast.
        type_list: List of *types*, aka ``CySparseType`` ``enum``\s.

    Warning:
        This function is **slow**.
    """
    if not (set(type_list) <= set(BASIC_TYPES)):
        raise TypeError('Som type(s) are not recognized as basic CySparseType')

    for type_el in type_list:

    
        if type_el == INT32_T:
            try:
                test_cast_to_INT32_t(n)
                return INT32_T
            except:
                pass
    

    
        elif type_el == UINT32_T:
            try:
                test_cast_to_UINT32_t(n)
                return UINT32_T
            except:
                pass
    

    
        elif type_el == INT64_T:
            try:
                test_cast_to_INT64_t(n)
                return INT64_T
            except:
                pass
    

    
        elif type_el == UINT64_T:
            try:
                test_cast_to_UINT64_t(n)
                return UINT64_T
            except:
                pass
    

    
        elif type_el == FLOAT32_T:
            try:
                test_cast_to_FLOAT32_t(n)
                return FLOAT32_T
            except:
                pass
    

    
        elif type_el == FLOAT64_T:
            try:
                test_cast_to_FLOAT64_t(n)
                return FLOAT64_T
            except:
                pass
    

    
        elif type_el == FLOAT128_T:
            try:
                test_cast_to_FLOAT128_t(n)
                return FLOAT128_T
            except:
                pass
    

    
        elif type_el == COMPLEX64_T:
            try:
                test_cast_to_COMPLEX64_t(n)
                return COMPLEX64_T
            except:
                pass
    

    
        elif type_el == COMPLEX128_T:
            try:
                test_cast_to_COMPLEX128_t(n)
                return COMPLEX128_T
            except:
                pass
    

    
        elif type_el == COMPLEX256_T:
            try:
                test_cast_to_COMPLEX256_t(n)
                return COMPLEX256_T
            except:
                pass
    


    raise TypeError('No type was found to cast number')


