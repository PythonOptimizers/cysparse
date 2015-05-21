import cysparse.types.cysparse_types as types

type1 = types.UINT32_T
type2 = types.INT32_T

print "type 1 is %s" % types.type_to_string(type1)
print types.string_to_type('INT32_t')
print types.INT32_T
print "Is type INT32_t really type INT32_t? " + str(types.string_to_type('INT32_t') == types.INT32_T)
print "Is type INT32_t an integer? " + str(types.is_integer(type2))
print "Is type FLOAT32_T an integer?" + str(types.is_integer(types.FLOAT32_T))

print "=" * 80

print "result types:"

for type1 in types.ELEMENT_TYPES:
    for type2 in types.ELEMENT_TYPES:
        print "(%s) + (%s) -> (%s)" % (types.type_to_string(type1), types.type_to_string(type2), types.type_to_string(types.result_type(type1, type2)))

print "=" * 80
print types.SIGNED_INTEGER_ELEMENT_TYPES
