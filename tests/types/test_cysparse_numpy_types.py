"""
Test CySparse's NumPy types conversions.
"""

from cysparse.types.cysparse_types import *
from cysparse.types.cysparse_numpy_types import *

import unittest

import numpy as np


class CySparseNumPyBaseTestCase(unittest.TestCase):
    def setUp(self):
        pass


class CySparseNumPyConversionTest(CySparseNumPyBaseTestCase):
    """
    Test basic conversions.
    """
    def test_to_cysparse_type(self):
        self.failUnless(numpy_to_cysparse_type(np.int32) == INT32_T)
        self.failUnless(numpy_to_cysparse_type(np.uint32) == UINT32_T)
        self.failUnless(numpy_to_cysparse_type(np.int64) == INT64_T)
        self.failUnless(numpy_to_cysparse_type(np.uint64) == UINT64_T)
        self.failUnless(numpy_to_cysparse_type(np.float32) == FLOAT32_T)
        self.failUnless(numpy_to_cysparse_type(np.float64) == FLOAT64_T)
        self.failUnless(numpy_to_cysparse_type(np.float128) == FLOAT128_T)
        self.failUnless(numpy_to_cysparse_type(np.complex64) == COMPLEX64_T)
        self.failUnless(numpy_to_cysparse_type(np.complex128) == COMPLEX128_T)
        self.failUnless(numpy_to_cysparse_type(np.complex256) == COMPLEX256_T)

    def test_to_numpy(self):
        self.failUnless(cysparse_to_numpy_type(INT32_T) ==np.int32)
        self.failUnless(cysparse_to_numpy_type(UINT32_T) == np.uint32)
        self.failUnless(cysparse_to_numpy_type(INT64_T) == np.int64)
        self.failUnless(cysparse_to_numpy_type(UINT64_T) == np.uint64)
        self.failUnless(cysparse_to_numpy_type(FLOAT32_T) == np.float32)
        self.failUnless(cysparse_to_numpy_type(FLOAT64_T) == np.float64)
        self.failUnless(cysparse_to_numpy_type(FLOAT128_T) == np.float128)
        self.failUnless(cysparse_to_numpy_type(COMPLEX64_T) == np.complex64)
        self.failUnless(cysparse_to_numpy_type(COMPLEX128_T) == np.complex128)
        self.failUnless(cysparse_to_numpy_type(COMPLEX256_T) == np.complex256)


class CySparseNumPyCompatibilityTest(CySparseNumPyBaseTestCase):
    """
    Test if two basic types are subtypes.
    """
    def test_to_cysparse_type(self):
        self.failUnless(are_mixed_types_compatible(INT32_T, np.int32) == True)
        self.failUnless(are_mixed_types_compatible(UINT32_T, np.uint32) == True)
        self.failUnless(are_mixed_types_compatible(INT64_T, np.int64) == True)
        self.failUnless(are_mixed_types_compatible(UINT64_T, np.uint64) == True)
        self.failUnless(are_mixed_types_compatible(FLOAT32_T, np.float32) == True)
        self.failUnless(are_mixed_types_compatible(FLOAT64_T, np.float64) == True)
        self.failUnless(are_mixed_types_compatible(FLOAT128_T, np.float128) == True)
        self.failUnless(are_mixed_types_compatible(COMPLEX64_T, np.complex64) == True)
        self.failUnless(are_mixed_types_compatible(COMPLEX128_T, np.complex128) == True)
        self.failUnless(are_mixed_types_compatible(COMPLEX256_T, np.complex256) == True)

    def test_numpy_compatibility(self):
        self.failUnless(is_numpy_type_compatible(np.int32) == True)
        self.failUnless(is_numpy_type_compatible(np.uint32) == True)
        self.failUnless(is_numpy_type_compatible(np.int64) ==  True)
        self.failUnless(is_numpy_type_compatible(np.uint64) == True)
        self.failUnless(is_numpy_type_compatible(np.float32) == True)
        self.failUnless(is_numpy_type_compatible(np.float64) == True)
        self.failUnless(is_numpy_type_compatible(np.float128) == True)
        self.failUnless(is_numpy_type_compatible(np.complex64) == True)
        self.failUnless(is_numpy_type_compatible(np.complex128) == True)
        self.failUnless(is_numpy_type_compatible(np.complex256) == True)

    def test_numpy_incompatibilities(self):
        """
        Some :program:`NumPy` incompatible types.


        """
        self.failUnless(is_numpy_type_compatible(np.int8) == False)
        self.failUnless(is_numpy_type_compatible(np.int16) == False)
        self.failUnless(is_numpy_type_compatible(np.uint8) == False)
        self.failUnless(is_numpy_type_compatible(np.uint16) == False)
        self.failUnless(is_numpy_type_compatible(np.float16) == False)

    def test_not_even_an_numpy_type(self):
        """
        Should raise a ``TypeError``.

        """
        with self.assertRaises(TypeError):
            numpy_to_cysparse_type(list)

if __name__ == '__main__':
    unittest.main()