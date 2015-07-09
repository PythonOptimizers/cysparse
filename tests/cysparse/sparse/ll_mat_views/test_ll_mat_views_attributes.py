#!/usr/bin/env python

"""
This file tests the attributes of an :class:`LLSparseMatrixView` object.

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
class CySparseLLSparseMatrixViewAttributesBaseTestCase(unittest.TestCase):
    def setUp(self):
        # DO NOT CHANGE THIS SETTING: tests are depending on it
        self.nrow = 4
        self.ncol = 6

        self.A1 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT64_T, row_wise=False)
        self.B1 = self.A1[:, :]
        self.B1_restrict = self.B1[0:4:2, [3, 2, 5]]
        self.B1_restrict_shape = (2, 3)

        self.A2 = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.nrow, itype=INT64_T, dtype=COMPLEX256_T,
                                              row_wise=True, store_zeros=True, is_symmetric=True)
        self.B2 = self.A2[:, :]
        self.B2_restrict = self.B2[0:4:2, [0, 1, 3]]
        self.B2_restrict_shape = (2, 3)


class CySparseLLSparseMatrixViewOutOfBoundsTestCase(CySparseLLSparseMatrixViewAttributesBaseTestCase):
    """
    Test base attributes.
    """
    def test_basic_attributes_for_view(self):
        self.failUnless(self.A1.nrow == self.B1.nrow == self.nrow)
        self.failUnless(self.A1.ncol == self.B1.ncol == self.ncol)

        self.failUnless(self.A1.shape == self.B1.shape)
        self.failUnless(self.A1.nnz == self.B1.nnz)

        self.failUnless(self.B1.type == 'LLSparseMatrixView')
        self.failUnless(self.B1.type_name == 'LLSparseMatrixView [INT32_t, FLOAT64_t]')

        self.failUnless(self.A1.dtype == self.B1.dtype == FLOAT64_T)
        self.failUnless(self.A1.itype == self.B1.itype == INT32_T)
        self.failUnless(self.A1.store_zeros == self.B1.store_zeros)
        self.failUnless(self.A1.is_symmetric == self.B1.is_symmetric)
        self.failUnless(self.A1.is_mutable == self.B1.is_mutable)
        self.failUnless(self.B1.is_empty is False)


        self.failUnless(self.A2.nrow == self.B2.nrow == self.nrow)
        self.failUnless(self.A2.ncol == self.B2.ncol == self.nrow)

        self.failUnless(self.A2.shape == self.B2.shape)
        self.failUnless(self.A2.nnz == self.B2.nnz)

        self.failUnless(self.B2.type == 'LLSparseMatrixView')
        self.failUnless(self.B2.type_name == 'LLSparseMatrixView [INT64_t, COMPLEX256_t]')
        self.failUnless(self.A2.dtype == self.B2.dtype == COMPLEX256_T)
        self.failUnless(self.A2.itype == self.B2.itype == INT64_T)
        self.failUnless(self.A2.store_zeros == self.B2.store_zeros)
        self.failUnless(self.A2.is_symmetric == self.B2.is_symmetric)
        self.failUnless(self.A2.is_mutable == self.B2.is_mutable)
        self.failUnless(self.B2.is_empty is False)

    def test_basic_attributes_for_view_of_view(self):
        self.failUnless(self.B1_restrict.nrow == self.B1_restrict_shape[0])
        self.failUnless(self.B1_restrict.ncol == self.B1_restrict_shape[1])

        self.failUnless(self.B1_restrict_shape == self.B1_restrict.shape)
        self.failUnless(self.B1_restrict.nnz == self.B1_restrict_shape[0] * self.B1_restrict_shape[1])

        self.failUnless(self.B1_restrict.type == 'LLSparseMatrixView')
        self.failUnless(self.B1_restrict.type_name == 'LLSparseMatrixView [INT32_t, FLOAT64_t]')

        self.failUnless(self.B1_restrict.dtype == self.B1.dtype == FLOAT64_T)
        self.failUnless(self.B1_restrict.itype == self.B1.itype == INT32_T)
        self.failUnless(self.B1_restrict.store_zeros == self.B1.store_zeros)
        self.failUnless(self.B1_restrict.is_symmetric == self.B1.is_symmetric)
        self.failUnless(self.B1_restrict.is_mutable == self.B1.is_mutable)
        self.failUnless(self.B1_restrict.is_empty is False)

        self.failUnless(self.B2_restrict.nrow == self.B2_restrict_shape[0])
        self.failUnless(self.B2_restrict.ncol == self.B2_restrict_shape[1])

        self.failUnless(self.B2_restrict_shape == self.B2_restrict.shape)
        self.failUnless(self.B2_restrict.nnz == 3)

        self.failUnless(self.B2_restrict.type == 'LLSparseMatrixView')
        self.failUnless(self.B2_restrict.type_name == 'LLSparseMatrixView [INT64_t, COMPLEX256_t]')

        self.failUnless(self.B2_restrict.dtype == self.B2.dtype == COMPLEX256_T)
        self.failUnless(self.B2_restrict.itype == self.B2.itype == INT64_T)
        self.failUnless(self.B2_restrict.store_zeros == self.B2.store_zeros)
        self.failUnless(self.B2_restrict.is_symmetric == self.B2.is_symmetric)
        self.failUnless(self.B2_restrict.is_mutable == self.B2.is_mutable)
        self.failUnless(self.B2_restrict.is_empty is False)


class CySparseLLSparseMatrixViewEmptinessTestCase(unittest.TestCase):
    """
    Test emptiness of views.
    """
    def setUp(self):
        # DO NOT CHANGE THIS SETTING: tests are depending on it
        self.nrow = 4
        self.ncol = 6

        self.A = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=INT32_T, dtype=FLOAT64_T, row_wise=False)
        self.B = self.A[:, :]
        self.B_restrict = self.B[0:4:2, [3, 2, 5]]
        self.B_restrict_shape = (2, 3)

        self.C = self.A[[1, 2], 7:8]  # empty
        self.D = self.B_restrict[4:5:7, [0]]  # empty

    def test_emptiness(self):
        self.failUnless(self.B.is_empty is False)
        self.failUnless(self.B_restrict.is_empty is False)
        self.failUnless(self.C.is_empty is True)
        self.failUnless(self.D.is_empty is True)

if __name__ == '__main__':
    unittest.main()
