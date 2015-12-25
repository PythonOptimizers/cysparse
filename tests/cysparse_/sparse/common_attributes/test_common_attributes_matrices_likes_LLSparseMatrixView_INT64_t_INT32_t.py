#!/usr/bin/env python

"""
This file tests basic common attributes for **all** matrix like objects.

Proxies are only tested for a :class:`LLSparseMatrix` object.

See file ``sparse_matrix_coherence_test_functions``.
"""
from sparse_matrix_like_common_attributes import common_matrix_like_attributes

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

##################################
# Case Non Symmetric, Non Zero
##################################
class CySparseCommonAttributesMatrices_LLSparseMatrixView_INT64_t_INT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14
        self.nnz = self.nrow * self.ncol
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT32_T, itype=INT64_T)

        self.C = self.A[:,:]
        self.base_type_str = 'LLSparseMatrixView'
        self.nargin = self.ncol
        self.nargout = self.nrow


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):

        self.assertTrue(self.C.nrow == self.nrow)


    def test_ncol_attribute(self):

        self.assertTrue(self.C.ncol == self.ncol)


    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz == self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(not self.C.store_zero)

    def test_is_mutable_attribute(self):

        self.assertTrue(self.C.is_mutable)


    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str, "'%s' is not '%s'" % (self.C.base_type_str, self.base_type_str))

    def test_is_symmetric(self):
        self.assertTrue(not self.C.is_symmetric)

    def test_nargin(self):
        self.assertTrue(self.nargin == self.C.nargin)

    def test_nargout(self):
        self.assertTrue(self.nargout == self.C.nargout)

##################################
# Case Symmetric, Non Zero
##################################
class CySparseCommonAttributesSymmetricMatrices_LLSparseMatrixView_INT64_t_INT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 14
        self.nnz = ((self.size + 1) * self.size) / 2
        self.nargin = self.size
        self.nargout = self.size

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=INT32_T, itype=INT64_T, store_symmetric=True)

        self.C = self.A[:,:]
        self.base_type_str = 'LLSparseMatrixView'


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):
        self.assertTrue(self.C.nrow == self.size)

    def test_ncol_attribute(self):
        self.assertTrue(self.C.ncol == self.size)

    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz == self.nnz, '%d is not %d' % (self.C.nnz, self.nnz))

    def test_symmetric_storage_attribute(self):
        self.assertTrue(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(not self.C.store_zero)

    def test_is_mutable_attribute(self):

        self.assertTrue(self.C.is_mutable)


    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)

    def test_is_symmetric(self):
        self.assertTrue(self.C.is_symmetric)

    def test_nargin(self):
        self.assertTrue(self.nargin == self.C.nargin)

    def test_nargout(self):
        self.assertTrue(self.nargout == self.C.nargout)

##################################
# Case Non Symmetric, Zero
##################################
class CySparseCommonAttributesWithZeroMatrices_LLSparseMatrixView_INT64_t_INT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14
        self.nnz = self.nrow * self.ncol
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT32_T, itype=INT64_T, store_zero=True)

        self.C = self.A[:,:]
        self.base_type_str = 'LLSparseMatrixView'
        self.nargin = self.ncol
        self.nargout = self.nrow


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):

        self.assertTrue(self.C.nrow == self.nrow)


    def test_ncol_attribute(self):

        self.assertTrue(self.C.ncol == self.ncol)


    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz == self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(self.C.store_zero)

    def test_is_mutable_attribute(self):

        self.assertTrue(self.C.is_mutable)


    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)

    def test_is_symmetric(self):
        self.assertTrue(not self.C.is_symmetric)

    def test_nargin(self):
        self.assertTrue(self.nargin == self.C.nargin)

    def test_nargout(self):
        self.assertTrue(self.nargout == self.C.nargout)

##################################
# Case Symmetric, Zero
##################################
class CySparseCommonAttributesSymmetricWithZeroMatrices_LLSparseMatrixView_INT64_t_INT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 14
        self.nnz = ((self.size + 1) * self.size) / 2
        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=INT32_T, itype=INT64_T, store_symmetric=True, store_zero=True)
        self.nargin = self.size
        self.nargout = self.size

        self.C = self.A[:,:]
        self.base_type_str = 'LLSparseMatrixView'


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):
        self.assertTrue(self.C.nrow == self.size)

    def test_ncol_attribute(self):
        self.assertTrue(self.C.ncol == self.size)

    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz, self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(self.C.store_zero)

    def test_is_mutable_attribute(self):

        self.assertTrue(self.C.is_mutable)


    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)

    def test_is_symmetric(self):
        self.assertTrue(self.C.is_symmetric)

    def test_nargin(self):
        self.assertTrue(self.nargin == self.C.nargin)

    def test_nargout(self):
        self.assertTrue(self.nargout == self.C.nargout)


if __name__ == '__main__':
    unittest.main()