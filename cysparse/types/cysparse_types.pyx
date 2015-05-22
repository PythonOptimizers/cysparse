from __future__ import print_function

cimport cysparse.types.cysparse_types as cp_types

from collections import OrderedDict
import sys

BASIC_TYPES_STR_DICT = OrderedDict()

# BASIC_TYPES_STR['type_str] = (Nbr of bits, enum value)
BASIC_TYPES_STR_DICT['INT32_t'] = (cp_types.INT32_t_BIT, cp_types.INT32_T)
BASIC_TYPES_STR_DICT['UINT32_t'] = (cp_types.UINT32_t_BIT, cp_types.UINT32_T)
BASIC_TYPES_STR_DICT['INT64_t'] = (cp_types.INT64_t_BIT, cp_types.INT64_T)
BASIC_TYPES_STR_DICT['UINT64_t'] = (cp_types.UINT64_t_BIT, cp_types.UINT64_T)
BASIC_TYPES_STR_DICT['FLOAT32_t'] = (cp_types.FLOAT32_t_BIT, cp_types.FLOAT32_T)
BASIC_TYPES_STR_DICT['FLOAT64_t'] = (cp_types.FLOAT64_t_BIT, cp_types.FLOAT64_T)
BASIC_TYPES_STR_DICT['COMPLEX64_t'] = (cp_types.COMPLEX64_t_BIT, cp_types.COMPLEX64_T)
BASIC_TYPES_STR_DICT['COMPLEX128_t'] = (cp_types.COMPLEX128_t_BIT, cp_types.COMPLEX128_T)

# construct inverse dict
# BASIC_TYPES_DICT[enum value] = (type string, nbr of bits)
BASIC_TYPES_DICT = {v[1]: (k, v[0]) for k, v in BASIC_TYPES_STR_DICT.items()}

# Type classification
# elements in general
ELEMENT_TYPES = BASIC_TYPES_DICT.keys()
INDEX_TYPES = [INT32_T, INT64_T]

# elements that behave like integers
INTEGER_ELEMENT_TYPES = [INT32_T, UINT32_T, INT64_T, UINT64_T]
UNSIGNED_INTEGER_ELEMENT_TYPES = [UINT32_T, UINT64_T]
SIGNED_INTEGER_ELEMENT_TYPES = [type1 for type1 in INTEGER_ELEMENT_TYPES if type1 not in UNSIGNED_INTEGER_ELEMENT_TYPES]

# elements that behave like real numbers (floats)
REAL_ELEMENT_TYPES = [FLOAT32_T, FLOAT64_T]
# elements that behave like complex numbers (we only consider complex floats)
COMPLEX_ELEMENT_TYPES = [COMPLEX64_T, COMPLEX128_T]

########################################################################################################################
# TESTS
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
            return True
    elif type2 == cp_types.COMPLEX64_T:
        if type1 in [cp_types.FLOAT32_T]:
            subtype = True
    elif type2 == cp_types.COMPLEX128_T:
        if type1 in [cp_types.FLOAT32_T, cp_types.FLOAT64_T, cp_types.COMPLEX64_T]:
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

# EXPLICIT TYPE TESTS
cpdef int result_type(cp_types.CySparseType type1, cp_types.CySparseType type2) except -1:
    """
    Return the resulting type between two types, i.e. the smallest compatible types with both types.

    Args:
        type1:
        type2:

    Returns:
        Resulting type.

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
        cp_types.CySparseType result_type, min_type, max_type

    min_type = min(type1, type2)
    max_type = max(type1, type2)

    result_type = max_type

    # CASE 1: same family type (integers, real or complex)
    if is_integer_type(min_type) and is_integer_type(max_type):
        if (is_signed_integer_type(min_type) and is_signed_integer_type(max_type)) or (is_unsigned_integer_type(min_type) and is_unsigned_integer_type(max_type)):
            result_type= max_type

        elif min_type == cp_types.INT32_T:
            if max_type in [cp_types.UINT32_T, cp_types.INT64_T]:
                result_type = cp_types.INT64_T
            elif max_type in [cp_types.UINT64_T]:
                raise TypeError("%s and %s are incompatible" % (type_to_string(min_type), type_to_string(max_type)))
            else:
                raise TypeError("Shouldn't happen. CODE 1. Please report.")
        elif min_type == cp_types.UINT32_T:
            if max_type in [cp_types.INT64_T]:
                result_type = cp_types.INT64_T
            elif max_type in [cp_types.UINT64_T]:
                result_type = cp_types.UINT64_T
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
        result_type = max_type
    elif is_complex_type(min_type) and is_complex_type(max_type):
        result_type = max_type

    # CASE 2: different family types
    # we use the default behavior of min_type/max_type and only consider the exceptions to this default behavior
    elif is_integer_type(min_type):
        if is_real_type(max_type):
            if min_type in [cp_types.INT64_T, cp_types.UINT64_T]:
                result_type = cp_types.FLOAT64_T
        elif is_complex_type(max_type):
            if min_type in [cp_types.INT64_T, cp_types.UINT64_T]:
                result_type = cp_types.COMPLEX128_T
    elif is_real_type(min_type):
        if is_complex_type(max_type):
            if min_type == cp_types.FLOAT64_T and max_type == cp_types.COMPLEX64_T:
                result_type = cp_types.COMPLEX128_T
    else:
        raise TypeError("Shouldn't happen. CODE 7. Please report.")

    return result_type


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
        print('{:12}: {:10d} bits ({:d})'.format(key, item[0], item[1]), file=OUT)


