cimport cysparse.types.cysparse_types as cp_types
cimport numpy as cnp
import numpy as np


def numpy_to_cysparse_type(numpy_type):
    if not np.issctype(numpy_type):
        raise TypeError("Not a NumPy type")

    if numpy_type == np.int32:
        return cp_types.INT32_T
    elif numpy_type == np.uint32:
        return cp_types.UINT32_T
    elif numpy_type == np.int64:
        return cp_types.INT64_T
    elif numpy_type == np.uint64:
        return cp_types.UINT64_T
    elif numpy_type == np.float32:
        return cp_types.FLOAT32_T
    elif numpy_type == np.float64:
        return cp_types.FLOAT64_T
    elif numpy_type == np.complex64:
        return cp_types.COMPLEX64_T
    elif numpy_type == np.complex128:
        return cp_types.COMPLEX128_T
    else:
        raise TypeError("Not a NumPy compatible type")

def is_numpy_type_compatible(numpy_type):
    try:
        numpy_to_cysparse_type(numpy_type)
        return True
    except TypeError:
        return False
