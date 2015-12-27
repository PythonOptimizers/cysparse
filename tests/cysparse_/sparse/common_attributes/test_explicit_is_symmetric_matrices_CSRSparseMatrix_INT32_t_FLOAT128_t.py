#!/usr/bin/env python

"""
This file tests the common attributes ``is_symmetric`` for **all** matrix objects.

``is_symmetric`` returns if a matrix or matrix-like object is indeed symmetric or not.

Warning:
    ``destroy_symmetry()`` uses randomness. Randomness should better be avoided in test cases but we do an exception
    here as it should not have any impact on the correctness of the tests.

Note:
    Don't be confused with ``store_symmetry``.
"""

import unittest
import random

from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

def restore_symmetry(A):
    """
    Restore symmetry in a square matrix.
    """
    ncol = A.ncol
    nrow = A.nrow

    assert ncol == nrow

    for i in range(nrow):
        for j in range(i):
            A[j, i] = A[i, j]

def destroy_symmetry(A):
    """
    Restore symmetry in a square matrix.
    """
    ncol = A.ncol
    nrow = A.nrow

    assert ncol == nrow

    i = random.randint(0, nrow - 1)
    j = random.randint(0, ncol - 1)
    while j == i:
        j = random.randint(0, ncol - 1)

    A[i, j] = A[j, i] + 1

########################################################################################################################
# Tests
########################################################################################################################
##################################
# Case Non Symmetric, Non Zero -> restore symmetry
##################################
class CySparseExplicitIsSymmetricAttributesRestoreSymmetryMatrices_CSRSparseMatrix_INT32_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=FLOAT128_T, itype=INT32_T)

        self.C = self.A.to_csr()


    def test_is_symmetric(self):
        self.assertTrue(not self.C.is_symmetric)

    def test_is_symmetric_after_change(self):
        restore_symmetry(self.A)

        self.C = self.A.to_csr()

        self.assertTrue(self.C.is_symmetric)

##################################
# Case Non Symmetric, Zero -> restore symmetry
##################################
class CySparseExplicitIsSymmetricAttributesRestoreSymmetryWithZeroMatrices_CSRSparseMatrix_INT32_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=FLOAT128_T, itype=INT32_T, store_zero=True)

        self.C = self.A.to_csr()


    def test_is_symmetric(self):
        self.assertTrue(not self.C.is_symmetric)

    def test_is_symmetric_after_change(self):
        restore_symmetry(self.A)

        self.C = self.A.to_csr()

        self.assertTrue(self.C.is_symmetric)


##################################
# Case Non Symmetric, Non Zero -> destroy symmetry
##################################
class CySparseExplicitIsSymmetricAttributesDestroySymmetryMatrices_CSRSparseMatrix_INT32_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = IdentityLLSparseMatrix(size=self.size, dtype=FLOAT128_T, itype=INT32_T)

        self.C = self.A.to_csr()


    def test_is_symmetric(self):
        self.assertTrue(self.C.is_symmetric)

    def test_is_symmetric_after_change(self):
        destroy_symmetry(self.A)

        self.C = self.A.to_csr()

        self.assertTrue(not self.C.is_symmetric)

##################################
# Case Non Symmetric, Zero -> destroy symmetry
##################################
class CySparseExplicitIsSymmetricAttributesDestroySymmetryWithZeroMatrices_CSRSparseMatrix_INT32_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = IdentityLLSparseMatrix(size=self.size, dtype=FLOAT128_T, itype=INT32_T, store_zero=True)

        self.C = self.A.to_csr()


    def test_is_symmetric(self):
        self.assertTrue(self.C.is_symmetric)

    def test_is_symmetric_after_change(self):
        destroy_symmetry(self.A)

        self.C = self.A.to_csr()

        self.assertTrue(not self.C.is_symmetric)

if __name__ == '__main__':
    unittest.main()