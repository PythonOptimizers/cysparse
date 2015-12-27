#!/usr/bin/env python

"""
This file tests factory methods for ``LLSparseMatrix``.

"""

import unittest
from cysparse.sparse.ll_mat import *


########################################################################################################################
# Tests
########################################################################################################################

NROW = 3
NCOL = 4
SIZE = 3


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseLLSparseMatrixFactoriesNoSymmetryNoZero_INT64_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL


    def test_LLSparseMatrix(self):
        self.A = LLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX64_T, itype=INT64_T)
        self.assertTrue(self.A.nnz == 0)


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseLLSparseMatrixFactoriesWithSymmetryNoZero_INT64_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

    def test_LLSparseMatrix(self):
        self.A = LLSparseMatrix(size=self.size, dtype=COMPLEX64_T, itype=INT64_T, store_symmetry=True)
        self.assertTrue(self.A.nnz == 0)

#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseLLSparseMatrixFactoriesNoSymmetrySWithZero_INT64_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

    def test_LLSparseMatrix(self):
        self.A = LLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=COMPLEX64_T, itype=INT64_T, store_zero=True)
        self.assertTrue(self.A.nnz == 0)


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseLLSparseMatrixFactoriesWithSymmetrySWithZero_INT64_t_COMPLEX64_t_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

    def test_LLSparseMatrix(self):
        self.A = LLSparseMatrix(size=self.size, dtype=COMPLEX64_T, itype=INT64_T, store_symmetry=True, store_zero=True)
        self.assertTrue(self.A.nnz == 0)

if __name__ == '__main__':
    unittest.main()
