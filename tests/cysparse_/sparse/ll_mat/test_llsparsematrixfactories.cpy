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
class CySparseLLSparseMatrixFactoriesNoSymmetryNoZero_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

    def test_LLSparseMatrix(self):

        self.A = LLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)
        #self.assertTrue()

    def test_XXX(self):
        pass


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseLLSparseMatrixFactoriesWithSymmetryNoZero_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

    def test_LLSparseMatrix(self):

        self.A = LLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True)


    def test_XXX(self):
        pass


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseLLSparseMatrixFactoriesNoSymmetrySWithZero_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

    def test_LLSparseMatrix(self):

        self.A = LLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@, store_zero=True)

    def test_XXX(self):
        pass



#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseLLSparseMatrixFactoriesWithSymmetrySWithZero_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

    def test_LLSparseMatrix(self):

        self.A = LLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True, store_zero=True)


    def test_XXX(self):
        pass


if __name__ == '__main__':
    unittest.main()

