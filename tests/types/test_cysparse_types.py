"""
Test CySparse's basic types.
"""

from cysparse.types.cysparse_types import *

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
        self.failUnless(is_subtype(INT32_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(INT32_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(UINT32_T, INT32_T) == False)
        self.failUnless(is_subtype(UINT32_T, UINT32_T) == True)
        self.failUnless(is_subtype(UINT32_T, INT64_T) == True)
        self.failUnless(is_subtype(UINT32_T, UINT64_T) == True)
        self.failUnless(is_subtype(UINT32_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(UINT32_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(UINT32_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(UINT32_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(INT64_T, INT32_T) == False)
        self.failUnless(is_subtype(INT64_T, UINT32_T) == False)
        self.failUnless(is_subtype(INT64_T, INT64_T) == True)
        self.failUnless(is_subtype(INT64_T, UINT64_T) == False)
        self.failUnless(is_subtype(INT64_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(INT64_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(INT64_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(INT64_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(UINT64_T, INT32_T) == False)
        self.failUnless(is_subtype(UINT64_T, UINT32_T) == False)
        self.failUnless(is_subtype(UINT64_T, INT64_T) == False)
        self.failUnless(is_subtype(UINT64_T, UINT64_T) == True)
        self.failUnless(is_subtype(UINT64_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(UINT64_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(UINT64_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(UINT64_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, INT32_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, UINT32_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, INT64_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, UINT64_T) == False)
        self.failUnless(is_subtype(FLOAT32_T, FLOAT32_T) == True)
        self.failUnless(is_subtype(FLOAT32_T, FLOAT64_T) == True)
        self.failUnless(is_subtype(FLOAT32_T, COMPLEX64_T) == True)
        self.failUnless(is_subtype(FLOAT32_T, COMPLEX128_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, INT32_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, UINT32_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, INT64_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, UINT64_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, FLOAT64_T) == True)
        self.failUnless(is_subtype(FLOAT64_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(FLOAT64_T, COMPLEX128_T) == True)
        self.failUnless(is_subtype(COMPLEX64_T, INT32_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, UINT32_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, INT64_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, UINT64_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(COMPLEX64_T, COMPLEX64_T) == True)
        self.failUnless(is_subtype(COMPLEX64_T, COMPLEX128_T) == True)
        self.failUnless(is_subtype(COMPLEX128_T, INT32_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, UINT32_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, INT64_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, UINT64_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, FLOAT32_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, FLOAT64_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, COMPLEX64_T) == False)
        self.failUnless(is_subtype(COMPLEX128_T, COMPLEX128_T) == True)

class CySparseTypesClassification(CySparseTypesBaseTestCase):
    def test_is_integer(self):
        self.failUnless(is_integer_type(INT32_T) == True)
        self.failUnless(is_integer_type(UINT32_T) == True)
        self.failUnless(is_integer_type(INT64_T) == True)
        self.failUnless(is_integer_type(UINT64_T) == True)
        self.failUnless(is_integer_type(FLOAT32_T) == False)
        self.failUnless(is_integer_type(FLOAT64_T) == False)
        self.failUnless(is_integer_type(COMPLEX64_T) == False)
        self.failUnless(is_integer_type(COMPLEX128_T) == False)

    def test_is_signed_integer(self):
        self.failUnless(is_signed_integer_type(INT32_T) == True)
        self.failUnless(is_signed_integer_type(UINT32_T) == False)
        self.failUnless(is_signed_integer_type(INT64_T) == True)
        self.failUnless(is_signed_integer_type(UINT64_T) == False)
        self.failUnless(is_signed_integer_type(FLOAT32_T) == False)
        self.failUnless(is_signed_integer_type(FLOAT64_T) == False)
        self.failUnless(is_signed_integer_type(COMPLEX64_T) == False)
        self.failUnless(is_signed_integer_type(COMPLEX128_T) == False)

    def test_is_unsigned_integer(self):
        self.failUnless(is_unsigned_integer_type(INT32_T) == False)
        self.failUnless(is_unsigned_integer_type(UINT32_T) == True)
        self.failUnless(is_unsigned_integer_type(INT64_T) == False)
        self.failUnless(is_unsigned_integer_type(UINT64_T) == True)
        self.failUnless(is_unsigned_integer_type(FLOAT32_T) == False)
        self.failUnless(is_unsigned_integer_type(FLOAT64_T) == False)
        self.failUnless(is_unsigned_integer_type(COMPLEX64_T) == False)
        self.failUnless(is_unsigned_integer_type(COMPLEX128_T) == False)

    def test_is_real_type(self):
        self.failUnless(is_real_type(INT32_T) == False)
        self.failUnless(is_real_type(UINT32_T) == False)
        self.failUnless(is_real_type(INT64_T) == False)
        self.failUnless(is_real_type(UINT64_T) == False)
        self.failUnless(is_real_type(FLOAT32_T) == True)
        self.failUnless(is_real_type(FLOAT64_T) == True)
        self.failUnless(is_real_type(COMPLEX64_T) == False)
        self.failUnless(is_real_type(COMPLEX128_T) == False)

    def test_is_complex_type(self):
        self.failUnless(is_complex_type(INT32_T) == False)
        self.failUnless(is_complex_type(UINT32_T) == False)
        self.failUnless(is_complex_type(INT64_T) == False)
        self.failUnless(is_complex_type(UINT64_T) == False)
        self.failUnless(is_complex_type(FLOAT32_T) == False)
        self.failUnless(is_complex_type(FLOAT64_T) == False)
        self.failUnless(is_complex_type(COMPLEX64_T) == True)
        self.failUnless(is_complex_type(COMPLEX128_T) == True)

    def test_is_index_type(self):
        self.failUnless(is_index_type(INT32_T) == True)
        self.failUnless(is_index_type(UINT32_T) == True)
        self.failUnless(is_index_type(INT64_T) == False)
        self.failUnless(is_index_type(UINT64_T) == False)
        self.failUnless(is_index_type(FLOAT32_T) == False)
        self.failUnless(is_index_type(FLOAT64_T) == False)
        self.failUnless(is_index_type(COMPLEX64_T) == False)
        self.failUnless(is_index_type(COMPLEX128_T) == False)

    def test_is_element_type(self):
        self.failUnless(is_element_type(INT32_T) == True)
        self.failUnless(is_element_type(UINT32_T) == True)
        self.failUnless(is_element_type(INT64_T) == True)
        self.failUnless(is_element_type(UINT64_T) == True)
        self.failUnless(is_element_type(FLOAT32_T) == True)
        self.failUnless(is_element_type(FLOAT64_T) == True)
        self.failUnless(is_element_type(COMPLEX64_T) == True)
        self.failUnless(is_element_type(COMPLEX128_T) == True)


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
        self.failUnless(result_type(INT32_T, COMPLEX64_T) == COMPLEX64_T)
        self.failUnless(result_type(INT32_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(UINT32_T, INT32_T) == INT64_T)
        self.failUnless(result_type(UINT32_T, UINT32_T) == UINT32_T)
        self.failUnless(result_type(UINT32_T, INT64_T) == INT64_T)
        self.failUnless(result_type(UINT32_T, UINT64_T) == UINT64_T)
        self.failUnless(result_type(UINT32_T, FLOAT32_T) == FLOAT32_T)
        self.failUnless(result_type(UINT32_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(UINT32_T, COMPLEX64_T) == COMPLEX64_T)
        self.failUnless(result_type(UINT32_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(INT64_T, INT32_T) == INT64_T)
        self.failUnless(result_type(INT64_T, UINT32_T) == INT64_T)
        self.failUnless(result_type(INT64_T, INT64_T) == INT64_T)
        self.failUnless(result_type(INT64_T, FLOAT32_T) == FLOAT64_T)
        self.failUnless(result_type(INT64_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(INT64_T, COMPLEX64_T) == COMPLEX128_T)
        self.failUnless(result_type(INT64_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(UINT64_T, UINT32_T) == UINT64_T)
        self.failUnless(result_type(UINT64_T, UINT64_T) == UINT64_T)
        self.failUnless(result_type(UINT64_T, FLOAT32_T) == FLOAT64_T)
        self.failUnless(result_type(UINT64_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(UINT64_T, COMPLEX64_T) == COMPLEX128_T)
        self.failUnless(result_type(UINT64_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(FLOAT32_T, INT32_T) == FLOAT32_T)
        self.failUnless(result_type(FLOAT32_T, UINT32_T) == FLOAT32_T)
        self.failUnless(result_type(FLOAT32_T, INT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT32_T, UINT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT32_T, FLOAT32_T) == FLOAT32_T)
        self.failUnless(result_type(FLOAT32_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT32_T, COMPLEX64_T) == COMPLEX64_T)
        self.failUnless(result_type(FLOAT32_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(FLOAT64_T, INT32_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, UINT32_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, INT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, UINT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, FLOAT32_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, FLOAT64_T) == FLOAT64_T)
        self.failUnless(result_type(FLOAT64_T, COMPLEX64_T) == COMPLEX128_T)
        self.failUnless(result_type(FLOAT64_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX64_T, INT32_T) == COMPLEX64_T)
        self.failUnless(result_type(COMPLEX64_T, UINT32_T) == COMPLEX64_T)
        self.failUnless(result_type(COMPLEX64_T, INT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX64_T, UINT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX64_T, FLOAT32_T) == COMPLEX64_T)
        self.failUnless(result_type(COMPLEX64_T, FLOAT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX64_T, COMPLEX64_T) == COMPLEX64_T)
        self.failUnless(result_type(COMPLEX64_T, COMPLEX128_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, INT32_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, UINT32_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, INT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, UINT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, FLOAT32_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, FLOAT64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, COMPLEX64_T) == COMPLEX128_T)
        self.failUnless(result_type(COMPLEX128_T, COMPLEX128_T) == COMPLEX128_T)

    def test_incompatible_result_type(self):
        with self.assertRaises(TypeError):
            result_type(INT32_T, UINT64_T)
            result_type(INT64_T, UINT64_T)
            result_type(UINT64_T, INT32_T)
            result_type(UINT64_T, INT64_T)


if __name__ == '__main__':
    unittest.main()