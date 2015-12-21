#!/usr/bin/env python

"""
This file tests basic common attributes for **all** matrix views objects  **except** :class:`TransposedSparseMatrix`.

See file ``sparse_matrix_coherence_test_functions``.
"""
from sparse_matrix_coherence_test_functions import common_matrix_like_attributes

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

class CySparseCommonAttributesMatricesViews_ConjugatedSparseMatrix_INT64_t_COMPLEX256_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=10, ncol=14, dtype=COMPLEX256_T, itype=INT64_T)

        self.C = self.A.conj


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_symmetric_storage_attribute(self):
        self.failUnless(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(not self.C.store_zero)


class CySparseCommonAttributesSymmetricMatricesViews_ConjugatedSparseMatrix_INT64_t_COMPLEX256_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=10, ncol=10, dtype=COMPLEX256_T, itype=INT64_T, store_symmetric=True)

        self.C = self.A.conj


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_symmetric_storage_attribute(self):
        self.failUnless(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(not self.C.store_zero)

class CySparseCommonAttributesWithZeroMatricesViews_ConjugatedSparseMatrix_INT64_t_COMPLEX256_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=10, ncol=14, dtype=COMPLEX256_T, itype=INT64_T, store_zero=True)

        self.C = self.A.conj


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_symmetric_storage_attribute(self):
        self.failUnless(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(self.C.store_zero)

class CySparseCommonAttributesSymmetricWithZeroMatricesViews_ConjugatedSparseMatrix_INT64_t_COMPLEX256_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=10, ncol=10, dtype=COMPLEX256_T, itype=INT64_T, store_symmetric=True, store_zero=True)

        self.C = self.A.conj


    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_symmetric_storage_attribute(self):
        self.failUnless(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(self.C.store_zero)

if __name__ == '__main__':
    unittest.main()