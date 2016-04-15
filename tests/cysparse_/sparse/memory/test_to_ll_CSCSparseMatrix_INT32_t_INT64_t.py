#!/usr/bin/env python

"""
This file tests ``to_ll`` for all matrices objects.

"""

import unittest
import numpy as np
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *


########################################################################################################################
# Tests
########################################################################################################################


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseToLLNoSymmetryNoZero_CSCSparseMatrix_INT32_t_INT64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT64_T, itype=INT32_T)


        self.C = self.A.to_csc()


    def test_to_ll_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        ll_mat = self.C.to_ll()

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i,j] == ll_mat[i, j])


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseToLLWithSymmetryNoZero_CSCSparseMatrix_INT32_t_INT64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=INT64_T, itype=INT32_T, store_symmetry=True)

        self.C = self.A.to_csc()


    def test_to_ll_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        ll_mat = self.C.to_ll()

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i,j] == ll_mat[i, j])


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseToLLNoSymmetrySWithZero_CSCSparseMatrix_INT32_t_INT64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT64_T, itype=INT32_T, store_zero=True)

        self.C = self.A.to_csc()


    def test_to_ll_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        ll_mat = self.C.to_ll()

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i,j] == ll_mat[i, j])

#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseToLLWithSymmetrySWithZero_CSCSparseMatrix_INT32_t_INT64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=INT64_T, itype=INT32_T, store_symmetry=True, store_zero=True)


        self.C = self.A.to_csc()


    def test_to_ll_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        ll_mat = self.C.to_ll()

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i,j] == ll_mat[i, j])


if __name__ == '__main__':
    unittest.main()
