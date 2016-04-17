from __future__ import print_function

cimport cysparse.common_types.cysparse_types as cp_types

cimport cython

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import sys

from cpython cimport PyObject

cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)
    int PyFloat_Check(PyObject *o)
    int PyComplex_Check(PyObject * o)
    int PyLong_Check(PyObject * o)
    int PyBool_Check(PyObject * o)

# export limits in Python for integer types only

INT32_t_MIN  = cp_types.INT32_MIN
INT32_t_MAX  = cp_types.INT32_MAX
UINT32_t_MAX = cp_types.UINT32_MAX
INT64_t_MIN  = cp_types.INT64_MIN
INT64_t_MAX  = cp_types.INT64_MAX
UINT64_t_MAX = cp_types.UINT64_MAX


BASIC_TYPES_STR_DICT = OrderedDict()

# BASIC_TYPES_STR['type_str] = (Nbr of bits, enum value, min value, max value)
# For real types: min value is the minimum representable floating-point number, max value the maximum representable floating-point number.
# For complex types: min value and max value correspond to the real types min/max values for real and imaginary parts.
BASIC_TYPES_STR_DICT['INT32_t'] = (cp_types.INT32_t_BIT, cp_types.INT32_T, cp_types.INT32_MIN, cp_types.INT32_MAX)
BASIC_TYPES_STR_DICT['UINT32_t'] = (cp_types.UINT32_t_BIT, cp_types.UINT32_T, 0, cp_types.UINT32_MAX)
BASIC_TYPES_STR_DICT['INT64_t'] = (cp_types.INT64_t_BIT, cp_types.INT64_T, cp_types.INT64_MIN, cp_types.INT64_MAX)
BASIC_TYPES_STR_DICT['UINT64_t'] = (cp_types.UINT64_t_BIT, cp_types.UINT64_T, 0, cp_types.UINT64_MAX)
BASIC_TYPES_STR_DICT['FLOAT32_t'] = (cp_types.FLOAT32_t_BIT, cp_types.FLOAT32_T, cp_types.FLT_MIN, cp_types.FLT_MAX)
BASIC_TYPES_STR_DICT['FLOAT64_t'] = (cp_types.FLOAT64_t_BIT, cp_types.FLOAT64_T, cp_types.DBL_MIN, cp_types.DBL_MAX)
BASIC_TYPES_STR_DICT['FLOAT128_t'] = (cp_types.FLOAT128_t_BIT, cp_types.FLOAT128_T, cp_types.LDBL_MIN, cp_types.LDBL_MAX)
BASIC_TYPES_STR_DICT['COMPLEX64_t'] = (cp_types.COMPLEX64_t_BIT, cp_types.COMPLEX64_T, cp_types.FLT_MIN, cp_types.FLT_MAX)
BASIC_TYPES_STR_DICT['COMPLEX128_t'] = (cp_types.COMPLEX128_t_BIT, cp_types.COMPLEX128_T, cp_types.DBL_MIN, cp_types.DBL_MAX)
BASIC_TYPES_STR_DICT['COMPLEX256_t'] = (cp_types.COMPLEX256_t_BIT, cp_types.COMPLEX256_T, cp_types.LDBL_MIN, cp_types.LDBL_MAX)

# construct inverse dict
# BASIC_TYPES_DICT[enum value] = (type string, nbr of bits, min value, max value)
BASIC_TYPES_DICT = {v[1]: (k, v[0], v[2], v[3]) for k, v in BASIC_TYPES_STR_DICT.items()}

# Type classification
# elements in general
BASIC_TYPES = BASIC_TYPES_DICT.keys()
ELEMENT_TYPES = BASIC_TYPES_DICT.keys()
INDEX_TYPES = [INT32_T, INT64_T]

# elements that behave like integers
INTEGER_ELEMENT_TYPES = [INT32_T, UINT32_T, INT64_T, UINT64_T]
UNSIGNED_INTEGER_ELEMENT_TYPES = [UINT32_T, UINT64_T]
SIGNED_INTEGER_ELEMENT_TYPES = [type1 for type1 in INTEGER_ELEMENT_TYPES if type1 not in UNSIGNED_INTEGER_ELEMENT_TYPES]

# elements that behave like real numbers (floats)
REAL_ELEMENT_TYPES = [FLOAT32_T, FLOAT64_T, FLOAT128_T]
# elements that behave like complex numbers (we only consider complex floats)
COMPLEX_ELEMENT_TYPES = [COMPLEX64_T, COMPLEX128_T, COMPLEX256_T]

cdef extern from "math.h":
    float INFINITY
    float NAN

inf = INFINITY
nan = NAN

########################################################################################################################
# TESTS ON TYPES
########################################################################################################################
# EXPLICIT TYPE TESTS
def is_subtype(cp_types.CySparseType type1, cp_types.CySparseType type2):
    """
    Tells if ``type1`` is a sub-type of ``type2`` for basic types.

    Returns:
        ``True`` if `type1`` is a sub-type of ``type2``, ``False`` otherwise.

    Args:
        type1: A ``CySparseType``.
        type2: Another ``CySparseType``.

    Note:
        Integers are **not** considered as a subtype of real numbers. Real numbers are considered as a subtype of
        complex numbers.
    """
    assert type1 in BASIC_TYPES_DICT and type2 in BASIC_TYPES_DICT, "Type(s) not recognized"

    subtype = False

    if type1 == type2:
        subtype = True
    elif type2 == cp_types.INT64_T:
        if type1 in [cp_types.UINT32_T, cp_types.INT32_T]:
            subtype = True
    elif type2 == cp_types.UINT64_T:
        if type1 in [cp_types.UINT32_T]:
            subtype = True
    elif type2 == cp_types.FLOAT64_T:
        if type1 in [cp_types.FLOAT32_T]:
            subtype = True
    elif type2 == cp_types.FLOAT128_T:
        if type1 in [cp_types.FLOAT32_T, cp_types.FLOAT64_T]:
            subtype = True
    elif type2 == cp_types.COMPLEX64_T:
        if type1 in [cp_types.FLOAT32_T]:
            subtype = True
    elif type2 == cp_types.COMPLEX128_T:
        if type1 in [cp_types.FLOAT32_T, cp_types.FLOAT64_T, cp_types.COMPLEX64_T]:
            subtype = True
    elif type2 == cp_types.COMPLEX256_T:
        if type1 in [cp_types.FLOAT32_T, cp_types.FLOAT64_T, cp_types.FLOAT128_T, cp_types.COMPLEX64_T, cp_types.COMPLEX128_T]:
            subtype = True

    return subtype


def is_integer_type(cp_types.CySparseType type1):
    """
    Return if type is integer.

    Args:
        type1:

    """
    return type1 in INTEGER_ELEMENT_TYPES

def is_signed_integer_type(cp_types.CySparseType type1):
    """
    Tell if type is signed integer type.

    Args:
        type1:
    """
    return type1 in SIGNED_INTEGER_ELEMENT_TYPES

def is_unsigned_integer_type(cp_types.CySparseType type1):
    """
    Tell if type is unsigned integer type.

    Args:
        type1:
    """
    return type1 in UNSIGNED_INTEGER_ELEMENT_TYPES

def is_real_type(cp_types.CySparseType type1):
    """
    Tell if type is a real.

    Args:
        type1:

    """
    return type1 in REAL_ELEMENT_TYPES


def is_complex_type(cp_types.CySparseType type1):
    """
    Tell if type is complex.

    Args:
        type1:

    """
    return type1 in COMPLEX_ELEMENT_TYPES

def is_index_type(cp_types.CySparseType type1):
    """
    Tell if type is indexable.

    Args:
        type1:
    """
    return type1 in INDEX_TYPES

def is_element_type(cp_types.CySparseType type1):
    """
    Tell if type can be used for matrix elements.

    Args:
        type1:

    """
    return type1 in ELEMENT_TYPES

########################################################################################################################
# TESTS ON NUMBERS
########################################################################################################################
cpdef is_python_number(object obj):
    """
    Test if an `obj` is recognized as a number by CPython.

    Args:
        obj:

    Warning:
        On 64bits architecture, `int32_t` is **not** recognized as an `int` by `PyInt_Check`. Use it if you **really**
        know what you are doing.
    """
    return PyInt_Check(<PyObject *>obj) or PyFloat_Check(<PyObject *>obj) or PyComplex_Check(<PyObject *>obj) or PyLong_Check(<PyObject *>obj) or PyBool_Check(<PyObject *>obj)

cpdef is_cysparse_number(obj):
    cdef:
        FLOAT128_t real_var
        COMPLEX256_t complex_var

    try:
        real_var = <FLOAT128_t> obj
        return True
    except:
        pass

    try:
        complex_var = <COMPLEX256_t> obj
        return True
    except:
        pass

    return False

cpdef is_scalar(obj):
    return is_python_number(obj) or is_cysparse_number(obj)

cpdef safe_cast_is_integer(obj):
    cdef INT64_t integer_var

    try:
        integer_var = <INT64_t> obj
        return True
    except:
        pass

    return False

########################################################################################################################
# TYPE CASTS
########################################################################################################################

# EXPLICIT TYPE TESTS
cpdef int result_type(cp_types.CySparseType type1, cp_types.CySparseType type2) except -1:
    """
    Return the resulting type between two types, i.e. the smallest compatible types with both types.

    Args:
        type1:
        type2:

    Returns:
        Resulting type. The return type is ``int`` to be compatible with ``Python``.

    Raises:
        ``TypeError`` whenever both types are **not** compatible.

    Warning:
        This function depends heavily on the explicit definition of basic types. You cannot add or remove a basic type
        **without** changing this function!
    """
    assert type1 in BASIC_TYPES_DICT and type2 in BASIC_TYPES_DICT, "Type(s) not recognized"

    if type1 == type2:
        return type1

    cdef:
        cp_types.CySparseType r_type, min_type, max_type

    min_type = min(type1, type2)
    max_type = max(type1, type2)

    r_type = max_type

    # CASE 1: same family type (integers, real or complex)
    if is_integer_type(min_type) and is_integer_type(max_type):
        if (is_signed_integer_type(min_type) and is_signed_integer_type(max_type)) or (is_unsigned_integer_type(min_type) and is_unsigned_integer_type(max_type)):
            r_type= max_type

        elif min_type == cp_types.INT32_T:
            if max_type in [cp_types.UINT32_T, cp_types.INT64_T]:
                r_type = cp_types.INT64_T
            elif max_type in [cp_types.UINT64_T]:
                raise TypeError("%s and %s are incompatible" % (type_to_string(min_type), type_to_string(max_type)))
            else:
                raise TypeError("Shouldn't happen. CODE 1. Please report.")
        elif min_type == cp_types.UINT32_T:
            if max_type in [cp_types.INT64_T]:
                r_type = cp_types.INT64_T
            elif max_type in [cp_types.UINT64_T]:
                r_type = cp_types.UINT64_T
            else:
                raise TypeError("Shouldn't happen. CODE 2. Please report.")
        elif min_type == cp_types.INT64_T:
            if max_type in [cp_types.UINT64_T]:
                raise TypeError("%s and %s are incompatible" % (type_to_string(min_type), type_to_string(max_type)))
            else:
                raise TypeError("Shouldn't happen. CODE 3. Please report.")
        else:
            raise TypeError("Shouldn't happen. CODE 4. Please report.")
    elif is_real_type(min_type) and is_integer_type(max_type):
        r_type = max_type
    elif is_complex_type(min_type) and is_complex_type(max_type):
        r_type = max_type

    # CASE 2: different family types
    # we use the default behavior of min_type/max_type and only consider the exceptions to this default behavior
    elif is_integer_type(min_type):
        if is_real_type(max_type):
            if min_type in [cp_types.INT64_T, cp_types.UINT64_T]:
                r_type = <cp_types.CySparseType> result_type(cp_types.FLOAT64_T, max_type)
        elif is_complex_type(max_type):
            if min_type in [cp_types.INT64_T, cp_types.UINT64_T]:
                r_type = <cp_types.CySparseType> result_type(cp_types.COMPLEX128_T, max_type)
    elif is_real_type(min_type):
        if is_complex_type(max_type):
            if min_type == cp_types.FLOAT64_T:
                r_type = <cp_types.CySparseType> result_type(cp_types.COMPLEX128_T, max_type)
            elif min_type == cp_types.FLOAT128_T:
                r_type = cp_types.COMPLEX256_T
            else:
                TypeError("Shouldn't happen. CODE 5. Please report.")
    elif is_complex_type(min_type):
        # both types are complex, max is OK
        pass
    else:
        raise TypeError("Shouldn't happen. CODE 6. Please report.")

    return r_type

cpdef int result_real_sum_type(cp_types.CySparseType type1):
    """
    Returns the best *real* type for a **real** sum for a given type.

    Args:
        type1:

    """
    cdef:
        cp_types.CySparseType r_type

    if type1 in [cp_types.INT32_T, cp_types.UINT32_T, cp_types.INT64_T, cp_types.UINT64_T]:
        r_type = cp_types.FLOAT64_T
    elif type1 in [cp_types.FLOAT32_T, cp_types.FLOAT64_T]:
        r_type = cp_types.FLOAT64_T
    elif type1 in [cp_types.FLOAT128_T]:
        r_type = cp_types.FLOAT128_T
    elif type1 in [cp_types.COMPLEX64_T, cp_types.COMPLEX128_T]:
        r_type = cp_types.FLOAT64_T
    elif type1 in [cp_types.COMPLEX256_T]:
        r_type = cp_types.FLOAT128_T
    else:
        raise TypeError("Not a recognized type")

    return r_type

########################################################################################################################
# I/O String/type conversions
########################################################################################################################
def type_to_string(cp_types.CySparseType type1):
    assert type1 in BASIC_TYPES_DICT, "Type not recognized"

    return BASIC_TYPES_DICT[type1][0]

def string_to_type(str type_string1):
    assert type_string1 in BASIC_TYPES_STR_DICT, "String type not recognized"

    return BASIC_TYPES_STR_DICT[type_string1][1]


########################################################################################################################
# REPORT
########################################################################################################################
def report_basic_types(OUT=sys.stdout):
    print('Basic types in CySparse:', file=OUT)
    print(file=OUT)
    for key, item in BASIC_TYPES_STR_DICT.items():
        print('{:12}: {:10d} bits ({:d})'.format(key, item[0], item[1]), file=OUT, end='')
        if BASIC_TYPES_STR_DICT[key][1] in INTEGER_ELEMENT_TYPES:
            print(" (min, max) = (", file=OUT, end='')
            print(item[2], file=OUT, end='')
            print(',', file=OUT, end='')
            print(item[3], file=OUT, end='')
            print(')', file=OUT)
        elif BASIC_TYPES_STR_DICT[key][1] in REAL_ELEMENT_TYPES:
            print(" (min precision, max precision) = (", file=OUT, end='')
            print(item[2], file=OUT, end='')
            print(',', file=OUT, end='')
            print(item[3], file=OUT, end='')
            print(')', file=OUT)
        elif BASIC_TYPES_STR_DICT[key][1] in COMPLEX_ELEMENT_TYPES:
            print(" (min precision real/imag, max precision real/imag) = (", file=OUT, end='')
            print(item[2], file=OUT, end='')
            print(',', file=OUT, end='')
            print(item[3], file=OUT, end='')
            print(')', file=OUT)


def report_CHAR_BIT(self):
    """
    Return the number of bits for 1 bytes on this platform.

    """
    return CHAR_BIT

########################################################################################################################
# NUMBER CASTS
########################################################################################################################
cpdef CySparseType min_integer_type(n, type_list) except? cp_types.UINT64_T:
    """
    Return the minimal type that can `n` can be casted into from a list of integer types.

    Note:
        We suppose that the list is sorted by ascending types.

    Raises:
        ``TypeError`` if no type can be used to cast `n` or if one element in the type list is not recognized as a
        ``CySparseType`` for an integer type.

    Args:
        n: Python number to cast.
        type_list: List of *types*, aka ``CySparseType`` ``enum``\s.

    Warning:
        This function is **slow**. There is no test on the ``n`` argument.
    """
    for type_el in type_list:
        if type_el in INTEGER_ELEMENT_TYPES:
            if BASIC_TYPES_DICT[type_el][2] <= n <= BASIC_TYPES_DICT[type_el][3]:
                return <cp_types.CySparseType> type_el
        else:
            raise TypeError('Type is not an integer type')

    raise TypeError('No type was found to cast number')

