from __future__ import print_function

cimport cysparse.types.cysparse_types as cp_types

from collections import OrderedDict
import sys



BASIC_TYPES_STR = OrderedDict()

# BASIC_TYPES_STR['type_str] = (Nbr of bits, enum value)
BASIC_TYPES_STR['INT32_t'] = (cp_types.INT32_t_BIT, cp_types.INT32_T)
BASIC_TYPES_STR['UINT32_t'] = (cp_types.UINT32_t_BIT, cp_types.UINT32_T)
BASIC_TYPES_STR['INT64_t'] = (cp_types.INT64_t_BIT, cp_types.INT64_T)
BASIC_TYPES_STR['UINT64_t'] = (cp_types.UINT64_t_BIT, cp_types.UINT64_T)
BASIC_TYPES_STR['FLOAT32_t'] = (cp_types.FLOAT32_t_BIT, cp_types.FLOAT32_T)
BASIC_TYPES_STR['FLOAT64_t'] = (cp_types.FLOAT64_t_BIT, cp_types.FLOAT64_T)
BASIC_TYPES_STR['COMPLEX64_t'] = (cp_types.COMPLEX64_t_BIT, cp_types.COMPLEX64_T)
BASIC_TYPES_STR['COMPLEX128_t'] = (cp_types.COMPLEX128_t_BIT, cp_types.COMPLEX128_T)

# construct inverse dict
BASIC_TYPES = {v[1]: (k, v[0]) for k, v in BASIC_TYPES_STR.items()}

ELEMENT_TYPES = BASIC_TYPES.keys()
INDEX_TYPES = [cp_types.COMPLEX64_T, cp_types.COMPLEX128_T]


########################################################################################################################
# TESTS
########################################################################################################################

########################################################################################################################
# REPORT
########################################################################################################################
def report_basic_types(OUT=sys.stdout):
    print('Basic types in CySparse:', file=OUT)
    print(file=OUT)
    for key, item in BASIC_TYPES_STR.items():
        print('{:12}: {:10d} bits ({:d})'.format(key, item[0], item[1]), file=OUT)


