import cysparse.cysparse_types.cysparse_types as types
import cysparse.cysparse_types.cysparse_numpy_types as np_types
import cysparse.sparse.ll_mat as ll_mat

import numpy as np

type1 = types.UINT32_T
type2 = types.INT32_T

print "type 1 is %s" % types.type_to_string(type1)
print types.string_to_type('INT32_t')
print types.INT32_T
print "Is type INT32_t really type INT32_t? " + str(types.string_to_type('INT32_t') == types.INT32_T)
print "Is type INT32_t an integer? " + str(types.is_integer_type(type2))
print "Is type FLOAT32_T an integer?" + str(types.is_integer_type(types.FLOAT32_T))

print "=" * 80

print "result types:"

for type1 in types.ELEMENT_TYPES:
    for type2 in types.ELEMENT_TYPES:
        print "(%s) + (%s):" % (types.type_to_string(type1), types.type_to_string(type2)),
        try:
            print " -> (%s)" % types.type_to_string(types.result_type(type1, type2))
        except TypeError as e:
            print
            print "    type error: " + str(e)

print "=" * 80
print types.SIGNED_INTEGER_ELEMENT_TYPES

types.report_basic_types()

print "=" * 80

type1 = types.INT32_T
type2 = types.FLOAT32_T
print "(%s) + (%s):" % (types.type_to_string(type1), types.type_to_string(type2)),
try:
    print " -> (%s)" % types.type_to_string(types.result_type(type1, type2))
except TypeError as e:
    print
    print "    type error: " + str(e)

print "=" * 80


for type1 in types.ELEMENT_TYPES:
    print "self.failUnless(is_element_type(%s) == %s)" % (types.type_to_string(type1), types.is_element_type(type1))

print "=" * 80


print "Numpy types"

a = np.array([2, 4, 5.8])
print a

print types.type_to_string(np_types.numpy_to_cysparse_type(a.dtype))

b = np.array([2, 4, 5.8], dtype=np_types.cysparse_to_numpy_type(types.UINT32_T))
print b
print b.dtype

print "=" * 80


print types.inf

print types.inf == np.inf
print types.nan
print types.nan + types.inf
print types.inf - types.inf

A = ll_mat.NewLLSparseMatrix(size=4, dtype=types.COMPLEX128_T)

A[0,0] = 3
A[1,0] = types.inf

import sys
A.print_to(sys.stdout)

print "=" * 80

B = ll_mat.NewLLSparseMatrix(size=2, dtype=types.FLOAT32_T)

B[0, 0] = 232
B[0, 1] = 1.3
B[1, 1] = 2**138
B[1, 0] = types.nan

print 2**137
B.print_to(sys.stdout)

print "=" * 80

B = ll_mat.NewLLSparseMatrix(size=2, dtype=types.INT64_T)

B[0, 0] = 232
B[0, 1] = 1.3
B[1, 1] = -0.89


B.print_to(sys.stdout)

C = B * B

C.print_to(sys.stdout)