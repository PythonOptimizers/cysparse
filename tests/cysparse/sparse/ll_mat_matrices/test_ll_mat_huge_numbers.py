"""
Test ``LLSparseMatrix`` and huge numbers.

The tests are **not** very clever and should **not** be used in a general test suite.

The results strongly depend on the system/compiler settings.

Floating-point math is hard!!!!!

"""

from cysparse.types.cysparse_types import *
from cysparse.sparse.ll_mat import *

import unittest

import sys

########################################################################################################################
# Tests
########################################################################################################################
class CySparseLLSparseMatrixHugeNumbersBaseTestCase(unittest.TestCase):
    def setUp(self):
        self.huge_number_float64 = 10e300
        self.huge_number_float128 = 10e308


        self.nan = nan
        self.inf = inf

        self.nbr_elements = 10
        self.size = 10

        self.A_float32 = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT32_T)
        self.A_float64 = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        self.A_float128 = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT128_T)

        self.A_complex64 = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=COMPLEX64_T)
        self.A_complex128 = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=COMPLEX128_T)
        self.A_complex256 = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=COMPLEX256_T)


class CySparseLLSparseMatrixHugeRealNumbersTestCase(CySparseLLSparseMatrixHugeNumbersBaseTestCase):
    """
    Test huge numbers for real matrices.
    """
    def test_huge_values_for_float32_matrices(self):

        self.A_float32[0, 0] = self.huge_number_float64
        self.A_float32[0, 1] = self.huge_number_float128

        self.failUnless(self.A_float32[0, 0] == self.inf)
        self.failUnless(self.A_float32[0, 1] == self.inf)

    def test_huge_values_for_float64_matrices(self):

        self.A_float64[0, 0] = self.huge_number_float64
        self.A_float64[0, 1] = self.huge_number_float128

        self.failUnless(self.A_float64[0, 0] != self.inf)
        self.failUnless(self.A_float64[0, 1] == self.inf)

    def test_huge_values_for_float128_matrices(self):

        self.A_float128[0, 0] = self.huge_number_float64
        self.A_float128[0, 1] = self.huge_number_float128

        self.failUnless(self.A_float128[0, 0] != self.inf)
        self.failUnless(self.A_float128[0, 1] == self.inf)

class CySparseLLSparseMatrixHugeComplexNumbersTestCase(CySparseLLSparseMatrixHugeNumbersBaseTestCase):
    """
    Test huge numbers for complex matrices.
    """
    def test_huge_values_for_complex64_matrices(self):

        self.A_complex64[0, 0] = self.huge_number_float64
        self.A_complex64[0, 1] = self.huge_number_float128

        self.A_complex64[0, 2] = self.huge_number_float64 + self.huge_number_float64 * 1j
        self.A_complex64[0, 3] = self.huge_number_float128 + self.huge_number_float128 * 1j


        self.failUnless(self.A_complex64[0, 0] == self.inf)
        self.failUnless(self.A_complex64[0, 1] == self.inf)

        self.failUnless(self.A_complex64[0, 2].imag == self.inf)
        self.failUnless(self.A_complex64[0, 3].imag == self.inf)

    def test_huge_values_for_complex128_matrices(self):

        self.A_complex128[0, 0] = self.huge_number_float64
        self.A_complex128[0, 1] = self.huge_number_float128

        self.A_complex128[0, 2] = self.huge_number_float64 + self.huge_number_float64 * 1j
        self.A_complex128[0, 3] = self.huge_number_float128 + self.huge_number_float128 * 1j

        self.failUnless(self.A_complex128[0, 0] != self.inf)
        self.failUnless(self.A_complex128[0, 1] == self.inf)

        self.failUnless(self.A_complex128[0, 2].imag != self.inf)
        self.failUnless(self.A_complex128[0, 3].imag == self.inf)

    def test_huge_values_for_complex256_matrices(self):

        self.A_complex256[0, 0] = self.huge_number_float64
        self.A_complex256[0, 1] = self.huge_number_float128

        self.A_complex256[0, 2] = self.huge_number_float64 + self.huge_number_float64 * 1j
        self.A_complex256[0, 3] = self.huge_number_float128 + self.huge_number_float128 * 1j

        self.failUnless(self.A_complex256[0, 0] != self.inf)
        self.failUnless(self.A_complex256[0, 1] == self.inf)

        self.failUnless(self.A_complex256[0, 2].imag != self.inf)
        self.failUnless(self.A_complex256[0, 3].imag == self.inf)

if __name__ == '__main__':
    unittest.main()

