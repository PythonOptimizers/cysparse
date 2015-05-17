from __future__ import print_function

from cysparse.cysparse_types cimport *

from collections import OrderedDict
import sys


BASIC_TYPES = OrderedDict()

# BASIC_TYPES['type] = (Nbr of bits, enum value)
BASIC_TYPES['INT32_t'] = (INT32_t_BIT, INT32_T)
BASIC_TYPES['UINT32_t'] = (UINT32_t_BIT, UINT32_T)
BASIC_TYPES['INT64_t'] = (INT64_t_BIT, INT64_T)
BASIC_TYPES['INT64_t'] = (INT64_t_BIT, INT64_T)
BASIC_TYPES['UINT64_t'] = (UINT64_t_BIT, UINT64_T)
BASIC_TYPES['FLOAT32_t'] = (FLOAT32_t_BIT, FLOAT32_T)
BASIC_TYPES['FLOAT64_t'] = (FLOAT64_t_BIT, FLOAT64_T)
BASIC_TYPES['COMPLEX64_t'] = (COMPLEX64_t_BIT, COMPLEX64_T)
BASIC_TYPES['COMPLEX128_t'] = (COMPLEX128_t_BIT, COMPLEX128_T)

_REVERSED_BASIC_TYPES = {v[1]: (k, v[0]) for k, v in BASIC_TYPES.items()}

# construct inverse dict

# BASIC_RELATIVE_TYPES['type] = (Nbr of bits, enum value)
BASIC_RELATIVE_TYPES = OrderedDict()
BASIC_RELATIVE_TYPES['SIZE_t'] = (SIZE_t_BIT, SIZE_T)
BASIC_RELATIVE_TYPES['INT_t'] = (INT_t_BIT, INT_T)
BASIC_RELATIVE_TYPES['FLOAT_t'] = (FLOAT_t_BIT, FLOAT_T)
BASIC_RELATIVE_TYPES['COMPLEX_t'] = (COMPLEX_t_BIT, COMPLEX_T)

_REVERSED_BASIC_RELATIVE_TYPES = {v[1]: (k, v[0]) for k, v in BASIC_RELATIVE_TYPES.items()}


########################################################################################################################
# TESTS
########################################################################################################################

########################################################################################################################
# REPORT
########################################################################################################################
def report_basic_types(OUT=sys.stdout):
    print('Basic types in CySparse:', file=OUT)
    print(file=OUT)
    for key, item in BASIC_TYPES.items():
        print('{:12}: {:10d} bits ({:d})'.format(key, item[0], item[1]), file=OUT)


def report_relative_basic_types(OUT=sys.stdout):
    print('Basic relative types in CySparse:', file=OUT)
    print(file=OUT)
    for key, item in BASIC_RELATIVE_TYPES.items():
        print('{:12}: {:10d} bits ({:d}) [{:12}]'.format(key, item[0], item[1], _REVERSED_BASIC_TYPES[item[1]][0]), file=OUT)

def report_types(OUT=sys.stdout):
    report_basic_types(OUT)
    print(file=OUT)
    report_relative_basic_types(OUT)