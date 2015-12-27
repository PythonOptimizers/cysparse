#!/usr/bin/env python

"""
This file tests ``to_ndarray`` for all matrices objects.

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
class CySparseToNDArrayNoSymmetryNoZero_CSCSparseMatrix_INT32_t_FLOAT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT32_T, itype=INT32_T)


        self.C = self.A.to_csc()


    def test_to_ndarray_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        ndarray = self.C.to_ndarray()

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i,j] == ndarray[i, j])


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseToNDArrayWithSymmetryNoZero_CSCSparseMatrix_INT32_t_FLOAT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=FLOAT32_T, itype=INT32_T, store_symmetry=True)

        self.C = self.A.to_csc()


    def test_to_ndarray_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        ndarray = self.C.to_ndarray()

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i,j] == ndarray[i, j])


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseToNDArrayNoSymmetrySWithZero_CSCSparseMatrix_INT32_t_FLOAT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT32_T, itype=INT32_T, store_zero=True)

        self.C = self.A.to_csc()


    def test_to_ndarray_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        ndarray = self.C.to_ndarray()

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i,j] == ndarray[i, j])

#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseToNDArrayWithSymmetrySWithZero_CSCSparseMatrix_INT32_t_FLOAT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=FLOAT32_T, itype=INT32_T, store_symmetry=True, store_zero=True)


        self.C = self.A.to_csc()


    def test_to_ndarray_element_by_element(self):
        nrow = self.C.nrow
        ncol = self.C.ncol

        ndarray = self.C.to_ndarray()

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i,j] == ndarray[i, j])


if __name__ == '__main__':
    unittest.main()
