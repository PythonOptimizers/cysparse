"""
Test CySparse's basic types.

We **only** test 64bits and 32bits architectures.
"""

from cysparse.common_types.cysparse_types import *

import unittest


class CySparseTypesBaseTestCase(unittest.TestCase):
    def setUp(self):
        pass


class CySparseTypesIsSubTypeTest(CySparseTypesBaseTestCase):
    """
    Test if two basic types are subtypes.
    """
    def test_is_subtype(self):
        self.failUnless(is_subtype(INT32_T, INT32_T) == True)
        self.failUnless(is_subtype(INT32_T, UINT32_T) == False)
        self.failUnless(is_subtype(INT32_T, INT64_T) == True)
        self.failUnless(is_subtype(INT32_T, UINT64_T) == False)
        self.failUnless(is_subtype(INT32_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(INT32_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(INT32_T, FLOAT128_T) == False)
        self.failUnless(is_subtype(INT32_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(INT32_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(INT32_T, COMPLEX256_T) == False)

        self.failUnless(is_subtype(UINT32_T, INT32_T) == False)
        self.failUnless(is_subtype(UINT32_T, UINT32_T) == True)
        self.failUnless(is_subtype(UINT32_T, INT64_T) == True)
        self.failUnless(is_subtype(UINT32_T, UINT64_T) == True)
        self.failUnless(is_subtype(UINT32_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(UINT32_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(UINT32_T, FLOAT128_T) == False)
        self.failUnless(is_subtype(UINT32_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(UINT32_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(UINT32_T, COMPLEX256_T) == False)

        self.failUnless(is_subtype(INT64_T, INT32_T) == False)
        self.failUnless(is_subtype(INT64_T, UINT32_T) == False)
        self.failUnless(is_subtype(INT64_T, INT64_T) == True)
        self.failUnless(is_subtype(INT64_T, UINT64_T) == False)
        self.failUnless(is_subtype(INT64_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(INT64_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(INT64_T, FLOAT128_T) == False)
        self.failUnless(is_subtype(INT64_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(INT64_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(INT64_T, COMPLEX256_T) == False)

        self.failUnless(is_subtype(UINT64_T, INT32_T) == False)
        self.failUnless(is_subtype(UINT64_T, UINT32_T) == False)
        self.failUnless(is_subtype(UINT64_T, INT64_T) == False)
        self.failUnless(is_subtype(UINT64_T, UINT64_T) == True)
        self.failUnless(is_subtype(UINT64_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(UINT64_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(UINT64_T, FLOAT128_T) == False)
        self.failUnless(is_subtype(UINT64_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(UINT64_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(UINT64_T, COMPLEX256_T) == False)

        self.failUnless(is_subtype(FLOAT32_T, INT32_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, UINT32_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, INT64_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, UINT64_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, FLOAT32_T) == True)
        self.failUnless(is_subtype(FLOAT32_T, FLOAT64_T) == True)
        self.failUnless(is_subtype(FLOAT32_T, FLOAT128_T) == True)
        self.failUnless(is_subtype(FLOAT32_T, COMPLEX64_T) == True)
        self.failUnless(is_subtype(FLOAT32_T, COMPLEX128_T) == True)
        self.failUnless(is_subtype(FLOAT32_T, COMPLEX256_T) == True)

        self.failUnless(is_subtype(FLOAT64_T, INT32_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, UINT32_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, INT64_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, UINT64_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, FLOAT64_T) == True)
        self.failUnless(is_subtype(FLOAT64_T, FLOAT128_T) == True)
        self.failUnless(is_subtype(FLOAT64_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, COMPLEX128_T) == True)
        self.failUnless(is_subtype(FLOAT64_T, COMPLEX256_T) == True)

        self.failUnless(is_subtype(FLOAT128_T, INT32_T) == False)
        self.failUnless(is_subtype(FLOAT128_T, UINT32_T) == False)
        self.failUnless(is_subtype(FLOAT128_T, INT64_T) == False)
        self.failUnless(is_subtype(FLOAT128_T, UINT64_T) == False)
        self.failUnless(is_subtype(FLOAT128_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(FLOAT128_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(FLOAT128_T, FLOAT128_T) == True)
        self.failUnless(is_subtype(FLOAT128_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(FLOAT128_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(FLOAT128_T, COMPLEX256_T) == True)

        self.failUnless(is_subtype(COMPLEX64_T, INT32_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, UINT32_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, INT64_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, UINT64_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, FLOAT128_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, COMPLEX64_T) == True)
        self.failUnless(is_subtype(COMPLEX64_T, COMPLEX128_T) == True)
        self.failUnless(is_subtype(COMPLEX64_T, COMPLEX256_T) == True)

        self.failUnless(is_subtype(COMPLEX128_T, INT32_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, UINT32_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, INT64_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, UINT64_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, FLOAT128_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, COMPLEX128_T) == True)
        self.failUnless(is_subtype(COMPLEX128_T, COMPLEX256_T) == True)


class CySparseTypesClassification(CySparseTypesBaseTestCase):
    def test_is_integer(self):
        self.failUnless(is_integer_type(INT32_T) == True)
        self.failUnless(is_integer_type(UINT32_T) == True)
        self.failUnless(is_integer_type(INT64_T) == True)
        self.failUnless(is_integer_type(UINT64_T) == True)
        self.failUnless(is_integer_type(FLOAT32_T) == False)
        self.failUnless(is_integer_type(FLOAT64_T) == False)
        self.failUnless(is_integer_type(FLOAT128_T) == False)
        self.failUnless(is_integer_type(COMPLEX64_T) == False)
        self.failUnless(is_integer_type(COMPLEX128_T) == False)
        self.failUnless(is_integer_type(COMPLEX256_T) == False)

    def test_is_signed_integer(self):
        self.failUnless(is_signed_integer_type(INT32_T) == True)
        self.failUnless(is_signed_integer_type(UINT32_T) == False)
        self.failUnless(is_signed_integer_type(INT64_T) == True)
        self.failUnless(is_signed_integer_type(UINT64_T) == False)
        self.failUnless(is_signed_integer_type(FLOAT32_T) == False)
        self.failUnless(is_signed_integer_type(FLOAT64_T) == False)
        self.failUnless(is_signed_integer_type(FLOAT128_T) == False)
        self.failUnless(is_signed_integer_type(COMPLEX64_T) == False)
        self.failUnless(is_signed_integer_type(COMPLEX128_T) == False)
        self.failUnless(is_signed_integer_type(COMPLEX256_T) == False)

    def test_is_unsigned_integer(self):
        self.failUnless(is_unsigned_integer_type(INT32_T) == False)
        self.failUnless(is_unsigned_integer_type(UINT32_T) == True)
        self.failUnless(is_unsigned_integer_type(INT64_T) == False)
        self.failUnless(is_unsigned_integer_type(UINT64_T) == True)
        self.failUnless(is_unsigned_integer_type(FLOAT32_T) == False)
        self.failUnless(is_unsigned_integer_type(FLOAT64_T) == False)
        self.failUnless(is_unsigned_integer_type(FLOAT128_T) == False)
        self.failUnless(is_unsigned_integer_type(COMPLEX64_T) == False)
        self.failUnless(is_unsigned_integer_type(COMPLEX128_T) == False)
        self.failUnless(is_unsigned_integer_type(COMPLEX256_T) == False)

    def test_is_real_type(self):
        self.failUnless(is_real_type(INT32_T) == False)
        self.failUnless(is_real_type(UINT32_T) == False)
        self.failUnless(is_real_type(INT64_T) == False)
        self.failUnless(is_real_type(UINT64_T) == False)
        self.failUnless(is_real_type(FLOAT32_T) == True)
        self.failUnless(is_real_type(FLOAT64_T) == True)
        self.failUnless(is_real_type(FLOAT128_T) == True)
        self.failUnless(is_real_type(COMPLEX64_T) == False)
        self.failUnless(is_real_type(COMPLEX128_T) == False)
        self.failUnless(is_real_type(COMPLEX256_T) == False)

    def test_is_complex_type(self):
        self.failUnless(is_complex_type(INT32_T) == False)
        self.failUnless(is_complex_type(UINT32_T) == False)
        self.failUnless(is_complex_type(INT64_T) == False)
        self.failUnless(is_complex_type(UINT64_T) == False)
        self.failUnless(is_complex_type(FLOAT32_T) == False)
        self.failUnless(is_complex_type(FLOAT64_T) == False)
        self.failUnless(is_complex_type(FLOAT128_T) == False)
        self.failUnless(is_complex_type(COMPLEX64_T) == True)
        self.failUnless(is_complex_type(COMPLEX128_T) == True)
        self.failUnless(is_complex_type(COMPLEX256_T) == True)

    def test_is_index_type(self):
        self.failUnless(is_index_type(INT32_T) == True)
        self.failUnless(is_index_type(UINT32_T) == False)
        self.failUnless(is_index_type(INT64_T) == True)
        self.failUnless(is_index_type(UINT64_T) == False)
        self.failUnless(is_index_type(FLOAT32_T) == False)
        self.failUnless(is_index_type(FLOAT64_T) == False)
        self.failUnless(is_index_type(FLOAT128_T) == False)
        self.failUnless(is_index_type(COMPLEX64_T) == False)
        self.failUnless(is_index_type(COMPLEX128_T) == False)
        self.failUnless(is_index_type(COMPLEX256_T) == False)

    def test_is_element_type(self):
        self.failUnless(is_element_type(INT32_T) == True)
        self.failUnless(is_element_type(UINT32_T) == True)
        self.failUnless(is_element_type(INT64_T) == True)
        self.failUnless(is_element_type(UINT64_T) == True)
        self.failUnless(is_element_type(FLOAT32_T) == True)
        self.failUnless(is_element_type(FLOAT64_T) == True)
        self.failUnless(is_element_type(FLOAT128_T) == True)
        self.failUnless(is_element_type(COMPLEX64_T) == True)
        self.failUnless(is_element_type(COMPLEX128_T) == True)
        self.failUnless(is_index_type(COMPLEX256_T) == False)


class CySparseTypesComparingBasicTypesTest(CySparseTypesBaseTestCase):
    """
    Test the resulting type given for two basic types.
    """
    def test_compatible_result_type(self):
        self.failUnless(result_type(INT32_T, INT32_T) == INT32_T)
        self.failUnless(result_type(INT32_T, UINT32_T) == INT64_T)
        self.failUnless(result_type(INT32_T, INT64_T) == INT64_T)
        self.failUnless(result_type(INT32_T, FLOAT32_T) == FLOAT32_T)
        self.failUnless(result_type(INT32_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(INT32_T, FLOAT128_T) == FLOAT128_T)
        self.failUnless(result_type(INT32_T, COMPLEX64_T) == COMPLEX64_T)
        self.failUnless(result_type(INT32_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(INT32_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(UINT32_T, INT32_T) == INT64_T)
        self.failUnless(result_type(UINT32_T, UINT32_T) == UINT32_T)
        self.failUnless(result_type(UINT32_T, INT64_T) == INT64_T)
        self.failUnless(result_type(UINT32_T, UINT64_T) == UINT64_T)
        self.failUnless(result_type(UINT32_T, FLOAT32_T) == FLOAT32_T)
        self.failUnless(result_type(UINT32_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(UINT32_T, FLOAT128_T) == FLOAT128_T)
        self.failUnless(result_type(UINT32_T, COMPLEX64_T) == COMPLEX64_T)
        self.failUnless(result_type(UINT32_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(UINT32_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(INT64_T, INT32_T) == INT64_T)
        self.failUnless(result_type(INT64_T, UINT32_T) == INT64_T)
        self.failUnless(result_type(INT64_T, INT64_T) == INT64_T)
        self.failUnless(result_type(INT64_T, FLOAT32_T) == FLOAT64_T)
        self.failUnless(result_type(INT64_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(INT64_T, FLOAT128_T) == FLOAT128_T)
        self.failUnless(result_type(INT64_T, COMPLEX64_T) == COMPLEX128_T)
        self.failUnless(result_type(INT64_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(INT64_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(UINT64_T, UINT32_T) == UINT64_T)
        self.failUnless(result_type(UINT64_T, UINT64_T) == UINT64_T)
        self.failUnless(result_type(UINT64_T, FLOAT32_T) == FLOAT64_T)
        self.failUnless(result_type(UINT64_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(UINT64_T, FLOAT128_T) == FLOAT128_T)
        self.failUnless(result_type(UINT64_T, COMPLEX64_T) == COMPLEX128_T)
        self.failUnless(result_type(UINT64_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(UINT64_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(FLOAT32_T, INT32_T) == FLOAT32_T)
        self.failUnless(result_type(FLOAT32_T, UINT32_T) == FLOAT32_T)
        self.failUnless(result_type(FLOAT32_T, INT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT32_T, UINT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT32_T, FLOAT32_T) == FLOAT32_T)
        self.failUnless(result_type(FLOAT32_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT32_T, FLOAT128_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT32_T, COMPLEX64_T) == COMPLEX64_T)
        self.failUnless(result_type(FLOAT32_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(FLOAT32_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(FLOAT64_T, INT32_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, UINT32_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, INT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, UINT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, FLOAT32_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, FLOAT128_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT64_T, COMPLEX64_T) == COMPLEX128_T)
        self.failUnless(result_type(FLOAT64_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(FLOAT64_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(FLOAT128_T, INT32_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT128_T, UINT32_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT128_T, INT64_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT128_T, UINT64_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT128_T, FLOAT32_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT128_T, FLOAT64_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT128_T, FLOAT128_T) == FLOAT128_T)
        self.failUnless(result_type(FLOAT128_T, COMPLEX64_T) == COMPLEX256_T)
        self.failUnless(result_type(FLOAT128_T, COMPLEX128_T) == COMPLEX256_T)
        self.failUnless(result_type(FLOAT128_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(COMPLEX64_T, INT32_T) == COMPLEX64_T)
        self.failUnless(result_type(COMPLEX64_T, UINT32_T) == COMPLEX64_T)
        self.failUnless(result_type(COMPLEX64_T, INT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX64_T, UINT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX64_T, FLOAT32_T) == COMPLEX64_T)
        self.failUnless(result_type(COMPLEX64_T, FLOAT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX64_T, FLOAT128_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX64_T, COMPLEX64_T) == COMPLEX64_T)
        self.failUnless(result_type(COMPLEX64_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX64_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(COMPLEX128_T, INT32_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, UINT32_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, INT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, UINT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, FLOAT32_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, FLOAT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, FLOAT128_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX128_T, COMPLEX64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, COMPLEX256_T) == COMPLEX256_T)

        self.failUnless(result_type(COMPLEX256_T, INT32_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, UINT32_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, INT64_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, UINT64_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, FLOAT32_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, FLOAT64_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, FLOAT128_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, COMPLEX64_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, COMPLEX128_T) == COMPLEX256_T)
        self.failUnless(result_type(COMPLEX256_T, COMPLEX256_T) == COMPLEX256_T)

    def test_incompatible_result_type(self):
        with self.assertRaises(TypeError):
            result_type(INT32_T, UINT64_T)
            result_type(INT64_T, UINT64_T)
            result_type(UINT64_T, INT32_T)
            result_type(UINT64_T, INT64_T)


class CySparseTypesNumberCastingBasicTypesTest(CySparseTypesBaseTestCase):
    """
    Test the resulting type corresponding to a number.
    """
    def setUp(self):
        import platform
        bits, linkage = platform.architecture()
        if bits == '64bit':
            self.unsigned_int32_n = 2**32 - 1
            self.unsigned_int64_n = 2**64 - 1

            self.signed_int32_n = 2**31-1
            self.signed_negative_int32_n = - 2**31 + 1

            self.signed_int64_n = 2**63-1
            self.signed_negative_int64_n = - 2**63 + 1

        elif bits == '32bit':
            self.unsigned_int32_n = 2**16 - 1
            self.unsigned_int64_n = 2**32 - 1

            self.signed_int32_n = 2**15-1
            self.signed_negative_int32_n = - 2**15 + 1

            self.signed_int64_n = 2**31-1
            self.signed_negative_int64_n = - 2**31 + 1

        self.big_integer = inf

    def test_minimal_unsigned_integer_type_for_a_number(self):
        self.failUnless(min_integer_type(self.unsigned_int32_n, UNSIGNED_INTEGER_ELEMENT_TYPES) == UINT32_T)
        self.failUnless(min_integer_type(self.unsigned_int64_n, UNSIGNED_INTEGER_ELEMENT_TYPES) == UINT64_T)

    def test_overflow_unsigned_integer_type_for_a_number(self):
        with self.assertRaises(TypeError):
            min_integer_type(self.big_integer, UNSIGNED_INTEGER_ELEMENT_TYPES)

    def test_minimal_signed_integer_type_for_a_number(self):
        self.failUnless(min_integer_type(self.signed_int32_n, SIGNED_INTEGER_ELEMENT_TYPES) == INT32_T)
        self.failUnless(min_integer_type(self.signed_int64_n, SIGNED_INTEGER_ELEMENT_TYPES) == INT64_T)

        self.failUnless(min_integer_type(self.signed_negative_int32_n, SIGNED_INTEGER_ELEMENT_TYPES) == INT32_T)
        self.failUnless(min_integer_type(self.signed_negative_int64_n, SIGNED_INTEGER_ELEMENT_TYPES) == INT64_T)

    def test_overflow_signed_integer_type_for_a_number(self):
        with self.assertRaises(TypeError):
            min_integer_type(self.big_integer, UNSIGNED_INTEGER_ELEMENT_TYPES)


class CySparseTypesNumberTestsTypesTest(CySparseTypesBaseTestCase):
    """
    Test numbers for type.
    """
    def setUp(self):

        self.an_int = 344
        self.a_real = 44.888
        self.a_complex = 45+54j
        self.a_list = [4, 5 , 6]
        self.a_string = "string"

    def test_is_python_number(self):
        self.failUnless(is_python_number(self.an_int))
        self.failUnless(is_python_number(self.a_real))
        self.failUnless(is_python_number(self.a_complex))

        self.failUnless(not is_python_number(self.a_list))
        self.failUnless(not is_python_number(self.a_string))

    def test_is_cysparse_number(self):
        self.failUnless(is_cysparse_number(self.an_int))
        self.failUnless(is_cysparse_number(self.a_real))
        self.failUnless(is_cysparse_number(self.a_complex))

        self.failUnless(not is_cysparse_number(self.a_list))
        self.failUnless(not is_cysparse_number(self.a_string))

    def test_is_scalar(self):
        self.failUnless(is_scalar(self.an_int))
        self.failUnless(is_scalar(self.a_real))
        self.failUnless(is_scalar(self.a_complex))

        self.failUnless(not is_scalar(self.a_list))
        self.failUnless(not is_scalar(self.a_string))


if __name__ == '__main__':
    unittest.main()