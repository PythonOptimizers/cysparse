#!/usr/bin/env python

"""
This file tests basic common attributes for **all** matrix like objects.

See file ``sparse_matrix_coherence_test_functions``.
"""
from sparse_matrix_coherence_test_functions import common_matrix_like_attributes

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

class CySparseCommonAttributesMatrices_CSCSparseMatrix_INT64_t_INT64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT64_T, itype=INT64_T)

        self.C = self.A.to_csc()
        self.nnz = self.nrow * self.ncol / 2 + min(self.nrow, self.ncol)
        self.base_type_str = 'CSCSparseMatrix'


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):
        self.assertTrue(self.C.nrow, self.nrow)

    def test_ncol_attribute(self):
        self.assertTrue(self.C.ncol, self.ncol)

    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz, self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(not self.C.store_zero)

    def test_is_mutable_attribute(self):

        self.assertTrue(not self.C.is_mutable)


    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)

class CySparseCommonAttributesSymmetricMatrices_CSCSparseMatrix_INT64_t_INT64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 14
        self.ncol = 14
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT64_T, itype=INT64_T, store_symmetric=True)

        self.C = self.A.to_csc()
        self.nnz = self.nrow * self.ncol / 2 + min(self.nrow, self.ncol)
        self.base_type_str = 'CSCSparseMatrix'


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):
        self.assertTrue(self.C.nrow, self.nrow)

    def test_ncol_attribute(self):
        self.assertTrue(self.C.ncol, self.ncol)

    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz, self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(not self.C.store_zero)

    def test_is_mutable_attribute(self):

        self.assertTrue(not self.C.is_mutable)


    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)


class CySparseCommonAttributesWithZeroMatrices_CSCSparseMatrix_INT64_t_INT64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT64_T, itype=INT64_T, store_zero=True)

        self.C = self.A.to_csc()
        self.nnz = self.nrow * self.ncol / 2 + min(self.nrow, self.ncol)
        self.base_type_str = 'CSCSparseMatrix'


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):
        self.assertTrue(self.C.nrow, self.nrow)

    def test_ncol_attribute(self):
        self.assertTrue(self.C.ncol, self.ncol)

    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz, self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(self.C.store_zero)

    def test_is_mutable_attribute(self):

        self.assertTrue(not self.C.is_mutable)


    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)


class CySparseCommonAttributesSymmetricWithZeroMatrices_CSCSparseMatrix_INT64_t_INT64_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 14
        self.ncol = 14
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT64_T, itype=INT64_T, store_symmetric=True, store_zero=True)

        self.C = self.A.to_csc()
        self.nnz = self.nrow * self.ncol / 2 + min(self.nrow, self.ncol)
        self.base_type_str = 'CSCSparseMatrix'


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):
        self.assertTrue(self.C.nrow, self.nrow)

    def test_ncol_attribute(self):
        self.assertTrue(self.C.ncol, self.ncol)

    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz, self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(self.C.store_zero)

    def test_is_mutable_attribute(self):

        self.assertTrue(not self.C.is_mutable)


    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)


if __name__ == '__main__':
    unittest.main()