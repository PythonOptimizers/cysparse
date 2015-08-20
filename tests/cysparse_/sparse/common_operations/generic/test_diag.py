#!/usr/bin/env python

"""
This file tests basic diagonals retrievals.

We test **all** types and the symmetric and general cases.


     If this is a python script (.py), it has been automatically generated by the 'generate_code.py' script.

"""
from cysparse.sparse.ll_mat import *
from cysparse.types.cysparse_types import *
import numpy as np

import unittest

import sys


def is_equal(A, B):
    if A.size != B.size:
        return False

    for i in xrange(A.size):
        if A[i] != B[i]:
            return False

    return True

########################################################################################################################
# Tests
########################################################################################################################
class CySparseCommonOperationsDiagonalsBaseTestCase(unittest.TestCase):
    def setUp(self):
        pass


class CySparseCommonOperationsDiagonalsTestCase(CySparseCommonOperationsDiagonalsBaseTestCase):
    """

    """
    def setUp(self):
        self.nbr_of_elements = 24
        self.nrow = 4
        self.ncol = 6


  
  
        self.l_1_1 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=INT32_T, row_wise=False)

        self.l_1_1_csc = self.l_1_1.to_csc()
        self.l_1_1_csr = self.l_1_1.to_csr()
  
        self.l_1_2 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=INT64_T, row_wise=False)

        self.l_1_2_csc = self.l_1_2.to_csc()
        self.l_1_2_csr = self.l_1_2.to_csr()
  
        self.l_1_3 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT32_T, row_wise=False)

        self.l_1_3_csc = self.l_1_3.to_csc()
        self.l_1_3_csr = self.l_1_3.to_csr()
  
        self.l_1_4 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT64_T, row_wise=False)

        self.l_1_4_csc = self.l_1_4.to_csc()
        self.l_1_4_csr = self.l_1_4.to_csr()
  
        self.l_1_5 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT128_T, row_wise=False)

        self.l_1_5_csc = self.l_1_5.to_csc()
        self.l_1_5_csr = self.l_1_5.to_csr()
  
        self.l_1_6 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=COMPLEX64_T, row_wise=False)

        self.l_1_6_csc = self.l_1_6.to_csc()
        self.l_1_6_csr = self.l_1_6.to_csr()
  
        self.l_1_7 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=COMPLEX128_T, row_wise=False)

        self.l_1_7_csc = self.l_1_7.to_csc()
        self.l_1_7_csr = self.l_1_7.to_csr()
  
        self.l_1_8 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=COMPLEX256_T, row_wise=False)

        self.l_1_8_csc = self.l_1_8.to_csc()
        self.l_1_8_csr = self.l_1_8.to_csr()
  

  
  
        self.l_2_1 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=INT32_T, row_wise=False)

        self.l_2_1_csc = self.l_2_1.to_csc()
        self.l_2_1_csr = self.l_2_1.to_csr()
  
        self.l_2_2 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=INT64_T, row_wise=False)

        self.l_2_2_csc = self.l_2_2.to_csc()
        self.l_2_2_csr = self.l_2_2.to_csr()
  
        self.l_2_3 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=FLOAT32_T, row_wise=False)

        self.l_2_3_csc = self.l_2_3.to_csc()
        self.l_2_3_csr = self.l_2_3.to_csr()
  
        self.l_2_4 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=FLOAT64_T, row_wise=False)

        self.l_2_4_csc = self.l_2_4.to_csc()
        self.l_2_4_csr = self.l_2_4.to_csr()
  
        self.l_2_5 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=FLOAT128_T, row_wise=False)

        self.l_2_5_csc = self.l_2_5.to_csc()
        self.l_2_5_csr = self.l_2_5.to_csr()
  
        self.l_2_6 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=COMPLEX64_T, row_wise=False)

        self.l_2_6_csc = self.l_2_6.to_csc()
        self.l_2_6_csr = self.l_2_6.to_csr()
  
        self.l_2_7 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=COMPLEX128_T, row_wise=False)

        self.l_2_7_csc = self.l_2_7.to_csc()
        self.l_2_7_csr = self.l_2_7.to_csr()
  
        self.l_2_8 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=COMPLEX256_T, row_wise=False)

        self.l_2_8_csc = self.l_2_8.to_csc()
        self.l_2_8_csr = self.l_2_8.to_csr()
  



    def test_simple_equality_one_by_one_negative_diagonals(self):

  
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_1_csc.diag(i), self.l_1_1_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_1_csr.diag(i), self.l_1_1.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_2_csc.diag(i), self.l_1_2_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_2_csr.diag(i), self.l_1_2.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_3_csc.diag(i), self.l_1_3_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_3_csr.diag(i), self.l_1_3.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_4_csc.diag(i), self.l_1_4_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_4_csr.diag(i), self.l_1_4.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_5_csc.diag(i), self.l_1_5_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_5_csr.diag(i), self.l_1_5.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_6_csc.diag(i), self.l_1_6_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_6_csr.diag(i), self.l_1_6.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_7_csc.diag(i), self.l_1_7_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_7_csr.diag(i), self.l_1_7.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_8_csc.diag(i), self.l_1_8_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_8_csr.diag(i), self.l_1_8.diag(i)))
  

  
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_1_csc.diag(i), self.l_2_1_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_1_csr.diag(i), self.l_2_1.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_2_csc.diag(i), self.l_2_2_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_2_csr.diag(i), self.l_2_2.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_3_csc.diag(i), self.l_2_3_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_3_csr.diag(i), self.l_2_3.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_4_csc.diag(i), self.l_2_4_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_4_csr.diag(i), self.l_2_4.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_5_csc.diag(i), self.l_2_5_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_5_csr.diag(i), self.l_2_5.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_6_csc.diag(i), self.l_2_6_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_6_csr.diag(i), self.l_2_6.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_7_csc.diag(i), self.l_2_7_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_7_csr.diag(i), self.l_2_7.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_8_csc.diag(i), self.l_2_8_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_8_csr.diag(i), self.l_2_8.diag(i)))
  


    def test_simple_equality_one_by_one_positive_diagonals(self):

  
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_1_csc.diag(i), self.l_1_1_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_1_csr.diag(i), self.l_1_1.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_2_csc.diag(i), self.l_1_2_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_2_csr.diag(i), self.l_1_2.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_3_csc.diag(i), self.l_1_3_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_3_csr.diag(i), self.l_1_3.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_4_csc.diag(i), self.l_1_4_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_4_csr.diag(i), self.l_1_4.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_5_csc.diag(i), self.l_1_5_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_5_csr.diag(i), self.l_1_5.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_6_csc.diag(i), self.l_1_6_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_6_csr.diag(i), self.l_1_6.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_7_csc.diag(i), self.l_1_7_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_7_csr.diag(i), self.l_1_7.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_8_csc.diag(i), self.l_1_8_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_8_csr.diag(i), self.l_1_8.diag(i)))
  

  
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_1_csc.diag(i), self.l_2_1_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_1_csr.diag(i), self.l_2_1.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_2_csc.diag(i), self.l_2_2_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_2_csr.diag(i), self.l_2_2.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_3_csc.diag(i), self.l_2_3_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_3_csr.diag(i), self.l_2_3.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_4_csc.diag(i), self.l_2_4_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_4_csr.diag(i), self.l_2_4.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_5_csc.diag(i), self.l_2_5_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_5_csr.diag(i), self.l_2_5.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_6_csc.diag(i), self.l_2_6_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_6_csr.diag(i), self.l_2_6.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_7_csc.diag(i), self.l_2_7_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_7_csr.diag(i), self.l_2_7.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_8_csc.diag(i), self.l_2_8_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_8_csr.diag(i), self.l_2_8.diag(i)))
  


class CySparseCommonOperationsSymDiagonalsTestCase(CySparseCommonOperationsDiagonalsBaseTestCase):
    """

    """
    def setUp(self):
        self.nbr_of_elements = 36
        self.nrow = 6
        self.ncol = 6


  
  
        self.l_1_1 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=INT32_T, is_symmetric=True, row_wise=False)

        self.l_1_1_csc = self.l_1_1.to_csc()
        self.l_1_1_csr = self.l_1_1.to_csr()
  
        self.l_1_2 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=INT64_T, is_symmetric=True, row_wise=False)

        self.l_1_2_csc = self.l_1_2.to_csc()
        self.l_1_2_csr = self.l_1_2.to_csr()
  
        self.l_1_3 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT32_T, is_symmetric=True, row_wise=False)

        self.l_1_3_csc = self.l_1_3.to_csc()
        self.l_1_3_csr = self.l_1_3.to_csr()
  
        self.l_1_4 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT64_T, is_symmetric=True, row_wise=False)

        self.l_1_4_csc = self.l_1_4.to_csc()
        self.l_1_4_csr = self.l_1_4.to_csr()
  
        self.l_1_5 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT128_T, is_symmetric=True, row_wise=False)

        self.l_1_5_csc = self.l_1_5.to_csc()
        self.l_1_5_csr = self.l_1_5.to_csr()
  
        self.l_1_6 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=COMPLEX64_T, is_symmetric=True, row_wise=False)

        self.l_1_6_csc = self.l_1_6.to_csc()
        self.l_1_6_csr = self.l_1_6.to_csr()
  
        self.l_1_7 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=COMPLEX128_T, is_symmetric=True, row_wise=False)

        self.l_1_7_csc = self.l_1_7.to_csc()
        self.l_1_7_csr = self.l_1_7.to_csr()
  
        self.l_1_8 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=COMPLEX256_T, is_symmetric=True, row_wise=False)

        self.l_1_8_csc = self.l_1_8.to_csc()
        self.l_1_8_csr = self.l_1_8.to_csr()
  

  
  
        self.l_2_1 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=INT32_T, is_symmetric=True, row_wise=False)

        self.l_2_1_csc = self.l_2_1.to_csc()
        self.l_2_1_csr = self.l_2_1.to_csr()
  
        self.l_2_2 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=INT64_T, is_symmetric=True, row_wise=False)

        self.l_2_2_csc = self.l_2_2.to_csc()
        self.l_2_2_csr = self.l_2_2.to_csr()
  
        self.l_2_3 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=FLOAT32_T, is_symmetric=True, row_wise=False)

        self.l_2_3_csc = self.l_2_3.to_csc()
        self.l_2_3_csr = self.l_2_3.to_csr()
  
        self.l_2_4 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=FLOAT64_T, is_symmetric=True, row_wise=False)

        self.l_2_4_csc = self.l_2_4.to_csc()
        self.l_2_4_csr = self.l_2_4.to_csr()
  
        self.l_2_5 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=FLOAT128_T, is_symmetric=True, row_wise=False)

        self.l_2_5_csc = self.l_2_5.to_csc()
        self.l_2_5_csr = self.l_2_5.to_csr()
  
        self.l_2_6 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=COMPLEX64_T, is_symmetric=True, row_wise=False)

        self.l_2_6_csc = self.l_2_6.to_csc()
        self.l_2_6_csr = self.l_2_6.to_csr()
  
        self.l_2_7 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=COMPLEX128_T, is_symmetric=True, row_wise=False)

        self.l_2_7_csc = self.l_2_7.to_csc()
        self.l_2_7_csr = self.l_2_7.to_csr()
  
        self.l_2_8 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT64_T, dtype=COMPLEX256_T, is_symmetric=True, row_wise=False)

        self.l_2_8_csc = self.l_2_8.to_csc()
        self.l_2_8_csr = self.l_2_8.to_csr()
  



    def test_simple_equality_one_by_one_negative_diagonals(self):

  
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_1_csc.diag(i), self.l_1_1_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_1_csr.diag(i), self.l_1_1.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_2_csc.diag(i), self.l_1_2_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_2_csr.diag(i), self.l_1_2.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_3_csc.diag(i), self.l_1_3_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_3_csr.diag(i), self.l_1_3.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_4_csc.diag(i), self.l_1_4_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_4_csr.diag(i), self.l_1_4.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_5_csc.diag(i), self.l_1_5_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_5_csr.diag(i), self.l_1_5.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_6_csc.diag(i), self.l_1_6_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_6_csr.diag(i), self.l_1_6.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_7_csc.diag(i), self.l_1_7_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_7_csr.diag(i), self.l_1_7.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_1_8_csc.diag(i), self.l_1_8_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_8_csr.diag(i), self.l_1_8.diag(i)))
  

  
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_1_csc.diag(i), self.l_2_1_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_1_csr.diag(i), self.l_2_1.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_2_csc.diag(i), self.l_2_2_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_2_csr.diag(i), self.l_2_2.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_3_csc.diag(i), self.l_2_3_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_3_csr.diag(i), self.l_2_3.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_4_csc.diag(i), self.l_2_4_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_4_csr.diag(i), self.l_2_4.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_5_csc.diag(i), self.l_2_5_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_5_csr.diag(i), self.l_2_5.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_6_csc.diag(i), self.l_2_6_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_6_csr.diag(i), self.l_2_6.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_7_csc.diag(i), self.l_2_7_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_7_csr.diag(i), self.l_2_7.diag(i)))
  
        # negative diagonals
        for i in xrange(0, -self.nrow, -1):
            self.failUnless(is_equal(self.l_2_8_csc.diag(i), self.l_2_8_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_8_csr.diag(i), self.l_2_8.diag(i)))
  


    def test_simple_equality_one_by_one_positive_diagonals(self):

  
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_1_csc.diag(i), self.l_1_1_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_1_csr.diag(i), self.l_1_1.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_2_csc.diag(i), self.l_1_2_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_2_csr.diag(i), self.l_1_2.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_3_csc.diag(i), self.l_1_3_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_3_csr.diag(i), self.l_1_3.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_4_csc.diag(i), self.l_1_4_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_4_csr.diag(i), self.l_1_4.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_5_csc.diag(i), self.l_1_5_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_5_csr.diag(i), self.l_1_5.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_6_csc.diag(i), self.l_1_6_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_6_csr.diag(i), self.l_1_6.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_7_csc.diag(i), self.l_1_7_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_7_csr.diag(i), self.l_1_7.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_1_8_csc.diag(i), self.l_1_8_csr.diag(i)))
            self.failUnless(is_equal(self.l_1_8_csr.diag(i), self.l_1_8.diag(i)))
  

  
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_1_csc.diag(i), self.l_2_1_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_1_csr.diag(i), self.l_2_1.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_2_csc.diag(i), self.l_2_2_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_2_csr.diag(i), self.l_2_2.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_3_csc.diag(i), self.l_2_3_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_3_csr.diag(i), self.l_2_3.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_4_csc.diag(i), self.l_2_4_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_4_csr.diag(i), self.l_2_4.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_5_csc.diag(i), self.l_2_5_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_5_csr.diag(i), self.l_2_5.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_6_csc.diag(i), self.l_2_6_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_6_csr.diag(i), self.l_2_6.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_7_csc.diag(i), self.l_2_7_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_7_csr.diag(i), self.l_2_7.diag(i)))
  
        # negative diagonals
        for i in xrange(1, self.ncol):
            self.failUnless(is_equal(self.l_2_8_csc.diag(i), self.l_2_8_csr.diag(i)))
            self.failUnless(is_equal(self.l_2_8_csr.diag(i), self.l_2_8.diag(i)))
  


if __name__ == '__main__':
    unittest.main()