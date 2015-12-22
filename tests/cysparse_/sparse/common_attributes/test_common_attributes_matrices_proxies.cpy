#!/usr/bin/env python

"""
This file tests basic common attributes for **all** matrix proxy.

See file ``sparse_matrix_coherence_test_functions``.
"""
from sparse_matrix_coherence_test_functions import common_matrix_like_attributes

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

class CySparseCommonAttributesMatricesViews_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)
{% if class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% else %}
YOU HAVE TO ADAPT THIS TEST TO A NEW MATRIX PROXY
{% endif %}

    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_symmetric_storage_attribute(self):
        self.failUnless(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(not self.C.store_zero)

    def test_nrow_attribute(self):
        self.assertTrue(self.C.nrow == self.nrow)

    def test_ncol_attribute(self):
        self.assertTrue(self.C.ncol == self.ncol)


class CySparseCommonAttributesSymmetricMatricesViews_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 14
        self.nnz = ((self.size + 1) * self.size) / 2

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetric=True)
{% if class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% else %}
YOU HAVE TO ADAPT THIS TEST TO A NEW MATRIX PROXY
{% endif %}

    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_symmetric_storage_attribute(self):
        self.failUnless(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(not self.C.store_zero)

class CySparseCommonAttributesWithZeroMatricesViews_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=10, ncol=14, dtype=@type|type2enum@, itype=@index|type2enum@, store_zero=True)
{% if class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% else %}
YOU HAVE TO ADAPT THIS TEST TO A NEW MATRIX PROXY
{% endif %}

    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_symmetric_storage_attribute(self):
        self.failUnless(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(self.C.store_zero)

class CySparseCommonAttributesSymmetricWithZeroMatricesViews_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=10, ncol=10, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetric=True, store_zero=True)
{% if class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% else %}
YOU HAVE TO ADAPT THIS TEST TO A NEW MATRIX PROXY
{% endif %}

    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_symmetric_storage_attribute(self):
        self.failUnless(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(self.C.store_zero)

if __name__ == '__main__':
    unittest.main()