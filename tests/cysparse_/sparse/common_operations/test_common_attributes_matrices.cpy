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

class CySparseCommonAttributesMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=10, ncol=14, dtype=@type|type2enum@, itype=@index|type2enum@)
{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% else %}
        self.C = self.A
{% endif %}

    def test_common_attributes(self):
        self.failUnless(common_matrix_like_attributes(self.C))

    def test_symmetric_storage_attribute(self):
        self.failUnless(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(not self.C.store_zero)


class CySparseCommonAttributesSymmetricMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=14, ncol=14, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetric=True)
{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% else %}
        self.C = self.A
{% endif %}

    def test_common_attributes(self):
        self.failUnless(common_matrix_like_attributes(self.C))

    def test_symmetric_storage_attribute(self):
        self.failUnless(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(not self.C.store_zero)


class CySparseCommonAttributesWithZeroMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=10, ncol=14, dtype=@type|type2enum@, itype=@index|type2enum@, store_zero=True)
{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% else %}
        self.C = self.A
{% endif %}

    def test_common_attributes(self):
        self.failUnless(common_matrix_like_attributes(self.C))

    def test_symmetric_storage_attribute(self):
        self.failUnless(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(self.C.store_zero)


class CySparseCommonAttributesSymmetricWithZeroMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.A = LinearFillLLSparseMatrix(nrow=14, ncol=14, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetric=True, store_zero=True)
{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% else %}
        self.C = self.A
{% endif %}

    def test_common_attributes(self):
        self.failUnless(common_matrix_like_attributes(self.C))

    def test_symmetric_storage_attribute(self):
        self.failUnless(self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.failUnless(self.C.store_zero)

if __name__ == '__main__':
    unittest.main()