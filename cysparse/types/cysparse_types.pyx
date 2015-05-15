from __future__ import print_function

from cysparse.types.cysparse_types cimport *

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

# construct inverse dict
_REVERSED_BASIC_TYPES = {v[1]: (k, v[0]) for k, v in BASIC_TYPES.items()}


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


