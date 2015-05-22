import cysparse.types.cysparse_types as types

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
