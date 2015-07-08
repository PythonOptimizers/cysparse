#!/usr/bin/env python

"""
This file tests the bounds of an :class:`LLSparseMatrixView` object.

Because the mechanism is the same for all :class:`LLSparseMatrixView` classes, we only test one case.

"""
from cysparse.sparse.ll_mat import *
from cysparse.types.cysparse_types import *
import numpy as np

import unittest

import sys


########################################################################################################################
# Tests
########################################################################################################################
class CySparseLLSparseMatrixBoundsBaseTestCase(unittest.TestCase):
    def setUp(self):
        pass


class CySparseLLSparseMatrixViewOutOfBoundsTestCase(CySparseLLSparseMatrixBoundsBaseTestCase):
    """
    Test out of bounds case.
    """
    def setUp(self):
        # DO NOT CHANGE THIS SETTING: tests are depending on it
        self.nrow = 4
        self.ncol = 6

        self.A = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT64_T, row_wise=False)
        self.B = self.A[:, :]
        self.B_restrict = self.A[0:4:2, [3,5]]
        self.B_restrict_shape = (2, 2)

    def test_simple_out_of_bound(self):
        with self.assertRaises(IndexError):
            temp = self.B[0, self.ncol]
            temp = self.B[self.nrow, 0]
            temp = self.B[-1, -1]
            temp = self.B[self.nrow, self.ncol]

    def test_simple_out_of_bound_restricted(self):
        with self.assertRaises(IndexError):
            temp = self.B_restrict[0, self.B_restrict_shape[1]]
            temp = self.B_restrict[self.B_restrict_shape[0], 0]
            temp = self.B_restrict[-1, -1]
            temp = self.B_restrict[self.nrow, self.ncol]

    def test_simple_in_bounds(self):
        for i in xrange(self.nrow):
            for j in xrange(self.ncol):
                self.failUnless(self.B[i, j] == self.A[i,j])

    def test_simple_in_bounds_restricted(self):
        self.failUnless(self.B_restrict.shape == self.B_restrict_shape)

        for i in xrange(self.B_restrict_shape[0]):
            for j in xrange(self.B_restrict_shape[1]):
                self.failUnless(self.B_restrict[i, j] == self.A[2*i,2*j+3])


class CySparseLLSparseMatrixViewOfLLSparseMatrixViewOutOfBoundsTestCase(CySparseLLSparseMatrixBoundsBaseTestCase):
    """
    Test out of bounds case for views of views.
    """
    def setUp(self):
        self.nrow = 4
        self.ncol = 6

        self.A = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT64_T, row_wise=False)
        self.B = self.A[:, :]
        self.C = self.B[:, :]
        self.C_restrict = self.B[0:4:2, [3,5]]
        self.C_restrict_shape = (2, 2)

    def test_simple_out_of_bound(self):
        with self.assertRaises(IndexError):
            temp = self.C[0, self.ncol]
            temp = self.C[self.nrow, 0]
            temp = self.C[-1, -1]
            temp = self.C[self.nrow, self.ncol]

    def test_simple_in_bounds(self):
        for i in xrange(self.nrow):
            for j in xrange(self.ncol):
                self.failUnless(self.B[i, j] == self.C[i,j])

    def test_simple_in_bounds_restricted(self):
        self.failUnless(self.C_restrict.shape == self.C_restrict_shape)

        for i in xrange(self.C_restrict_shape[0]):
            for j in xrange(self.C_restrict_shape[1]):
                self.failUnless(self.C_restrict[i, j] == self.C[2*i,2*j+3])

if __name__ == '__main__':
    unittest.main()