#!/usr/bin/env python

"""
This file tests upper and lower triangular sub-matrices for all matrices objects.

"""

import unittest
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
class CySparseTriangularNoSymmetryNoZero_CSCSparseMatrix_INT64_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX64_T, itype=INT64_T)


        self.C = self.A.to_csc()


        self.C_tril = self.C.tril()

    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        max_range = min(nrow, ncol)

        for i in range(nrow):
            for j in range(i + 1):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

                if j == max_range:
                    break


    def test_triu_default(self):
        pass


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseTriangularWithSymmetryNoZero_CSCSparseMatrix_INT64_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX64_T, itype=INT64_T, store_symmetry=True)


        self.C = self.A.to_csc()


        self.C_tril = self.C.tril()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        max_range = min(nrow, ncol)

        for i in range(nrow):
            for j in range(i + 1):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

                if j == max_range:
                    break


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseTriangularNoSymmetrySWithZero_CSCSparseMatrix_INT64_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX64_T, itype=INT64_T, store_zero=True)


        self.C = self.A.to_csc()


        self.C_tril = self.C.tril()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        max_range = min(nrow, ncol)

        for i in range(nrow):
            for j in range(i + 1):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

                if j == max_range:
                    break

#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseTriangularWithSymmetrySWithZero_CSCSparseMatrix_INT64_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=COMPLEX64_T, itype=INT64_T, store_symmetry=True, store_zero=True)


        self.C = self.A.to_csc()


        self.C_tril = self.C.tril()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        max_range = min(nrow, ncol)

        for i in range(nrow):
            for j in range(i + 1):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

                if j == max_range:
                    break

if __name__ == '__main__':
    unittest.main()
