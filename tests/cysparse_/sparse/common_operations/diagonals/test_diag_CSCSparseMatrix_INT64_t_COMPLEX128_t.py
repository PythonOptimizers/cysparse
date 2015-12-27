#!/usr/bin/env python

"""
This file tests ``diag()`` for all matrices objects.

"""

import unittest
import numpy as np
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *


########################################################################################################################
# Tests
########################################################################################################################
NROW = 10
NCOL = 14
SIZE = 10

#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseDiagNoSymmetryNoZero_CSCSparseMatrix_INT64_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX128_T, itype=INT64_T)


        self.C = self.A.to_csc()


    def test_default_diag_elment_by_element(self):
        """
        Test ``diag()`` with default argument.

        By default, ``diag()`` returns the main diagonal.
        """
        diag = self.C.diag()

        diag_length = min(self.C.nrow, self.ncol)

        for k in range(diag_length):
            self.assertTrue(diag[k] == self.A[k, k])

    def test_negative_diag_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(-nrow + 1, 0, -1):
            diag_length = min(nrow+k, ncol)

            diag = self.C.diag(k)
            for j in range(diag_length):
                self.assertTrue(diag[j] == self.A[j - k, j])

    def test_positive_diag_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(1, ncol, 1):
            diag_length = min(nrow, ncol - k)

            diag = self.C.diag(k)
            for j in range(diag_length):
                self.assertTrue(diag[j] == self.A[j, j + k])



#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseDiagWithSymmetryNoZero_CSCSparseMatrix_INT64_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX128_T, itype=INT64_T, store_symmetry=True)


        self.C = self.A.to_csc()


    def test_default_diag_elment_by_element(self):
        """
        Test ``diag()`` with default argument.

        By default, ``diag()`` returns the main diagonal.
        """
        diag = self.C.diag()

        diag_length = self.size

        for k in range(diag_length):
            self.assertTrue(diag[k] == self.A[k, k])

    def test_negative_diag_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(-nrow + 1, 0, -1):
            diag_length = min(nrow+k, ncol)

            diag = self.C.diag(k)
            for j in range(diag_length):
                self.assertTrue(diag[j] == self.A[j - k, j])

    def test_positive_diag_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(1, ncol, 1):
            diag_length = min(nrow, ncol - k)

            diag = self.C.diag(k)
            for j in range(diag_length):
                self.assertTrue(diag[j] == self.A[j, j + k])


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseDiagNoSymmetrySWithZero_CSCSparseMatrix_INT64_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX128_T, itype=INT64_T, store_zero=True)


        self.C = self.A.to_csc()


    def test_default_diag_elment_by_element(self):
        """
        Test ``diag()`` with default argument.

        By default, ``diag()`` returns the main diagonal.
        """
        diag = self.C.diag()

        diag_length = min(self.C.nrow, self.ncol)

        for k in range(diag_length):
            self.assertTrue(diag[k] == self.A[k, k])

    def test_negative_diag_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(-nrow + 1, 0, -1):
            diag_length = min(nrow+k, ncol)

            diag = self.C.diag(k)
            for j in range(diag_length):
                self.assertTrue(diag[j] == self.A[j - k, j])

    def test_positive_diag_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(1, ncol, 1):
            diag_length = min(nrow, ncol - k)

            diag = self.C.diag(k)
            for j in range(diag_length):
                self.assertTrue(diag[j] == self.A[j, j + k])


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseDiagWithSymmetrySWithZero_CSCSparseMatrix_INT64_t_COMPLEX128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX128_T, itype=INT64_T, store_symmetry=True, store_zero=True)


        self.C = self.A.to_csc()


    def test_default_diag_elment_by_element(self):
        """
        Test ``diag()`` with default argument.

        By default, ``diag()`` returns the main diagonal.
        """
        diag = self.C.diag()

        diag_length = self.size

        for k in range(diag_length):
            self.assertTrue(diag[k] == self.A[k, k])

    def test_negative_diag_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(-nrow + 1, 0, -1):
            diag_length = min(nrow+k, ncol)

            diag = self.C.diag(k)
            for j in range(diag_length):
                self.assertTrue(diag[j] == self.A[j - k, j])

    def test_positive_diag_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(1, ncol, 1):
            diag_length = min(nrow, ncol - k)

            diag = self.C.diag(k)
            for j in range(diag_length):
                self.assertTrue(diag[j] == self.A[j, j + k])


if __name__ == '__main__':
    unittest.main()
