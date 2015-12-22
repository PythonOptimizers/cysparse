#!/usr/bin/env python

"""
This file tests basic common attributes for **all** matrix like objects.

Proxies are only tested for a :class:`LLSparseMatrix` object.

See file ``sparse_matrix_coherence_test_functions``.
"""
from sparse_matrix_coherence_test_functions import common_matrix_like_attributes

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

##################################
# Case Non Symmetric, Non Zero
##################################
class CySparseCommonAttributesMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14
        self.nnz = self.nrow * self.ncol
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)
{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
        self.base_type_str = 'CSCSparseMatrix'
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
        self.base_type_str = 'CSRSparseMatrix'
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
        self.base_type_str = 'LLSparseMatrix'
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
        self.base_type_str = 'Transposed of ' + self.A.base_type_str
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% if type in complex_list %}
        self.base_type_str = 'Conjugated of ' + self.A.base_type_str
{% else %}
        self.base_type_str = self.A.base_type_str
{% endif %}
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% if type in complex_list %}
        self.base_type_str = 'Conjugate Transposed of ' + self.A.base_type_str
{% else %}
        self.base_type_str = 'Transposed of ' + self.A.base_type_str
{% endif %}
{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]
        self.base_type_str = 'LLSparseMatrixView'
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
        self.assertTrue(self.C.nrow == self.ncol)
{% else %}
        self.assertTrue(self.C.nrow == self.nrow)
{% endif %}

    def test_ncol_attribute(self):
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
        self.assertTrue(self.C.ncol == self.nrow)
{% else %}
        self.assertTrue(self.C.ncol == self.ncol)
{% endif %}

    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz == self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(not self.C.store_zero)

    def test_is_mutable_attribute(self):
{% if class == 'CSCSparseMatrix' %}
        self.assertTrue(not self.C.is_mutable)
{% elif class == 'CSRSparseMatrix' %}
        self.assertTrue(not self.C.is_mutable)
{% elif class == 'LLSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'TransposedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'ConjugatedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'LLSparseMatrixView' %}
        self.assertTrue(self.C.is_mutable)
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str, "'%s' is not '%s'" % (self.C.base_type_str, self.base_type_str))

    def test_is_symmetric(self):
        self.assertTrue(not self.C.is_symmetric)

##################################
# Case Symmetric, Non Zero
##################################
class CySparseCommonAttributesSymmetricMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 14
        self.nnz = ((self.size + 1) * self.size) / 2

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetric=True)
{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
        self.base_type_str = 'CSCSparseMatrix'
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
        self.base_type_str = 'CSRSparseMatrix'
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
        self.base_type_str = 'LLSparseMatrix'
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
        self.base_type_str = 'Transposed of ' + self.A.base_type_str
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% if type in complex_list %}
        self.base_type_str = 'Conjugated of ' + self.A.base_type_str
{% else %}
        self.base_type_str = self.A.base_type_str
{% endif %}
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% if type in complex_list %}
        self.base_type_str = 'Conjugate Transposed of ' + self.A.base_type_str
{% else %}
        self.base_type_str = 'Transposed of ' + self.A.base_type_str
{% endif %}
{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]
        self.base_type_str = 'LLSparseMatrixView'
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

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
{% if class == 'CSCSparseMatrix' %}
        self.assertTrue(not self.C.is_mutable)
{% elif class == 'CSRSparseMatrix' %}
        self.assertTrue(not self.C.is_mutable)
{% elif class == 'LLSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'TransposedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'ConjugatedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'LLSparseMatrixView' %}
        self.assertTrue(self.C.is_mutable)
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)

    def test_is_symmetric(self):
        self.assertTrue(self.C.is_symmetric)

##################################
# Case Non Symmetric, Zero
##################################
class CySparseCommonAttributesWithZeroMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14
        self.nnz = self.nrow * self.ncol
        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@, store_zero=True)
{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
        self.base_type_str = 'CSCSparseMatrix'
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
        self.base_type_str = 'CSRSparseMatrix'
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
        self.base_type_str = 'LLSparseMatrix'
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
        self.base_type_str = 'Transposed of ' + self.A.base_type_str
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% if type in complex_list %}
        self.base_type_str = 'Conjugated of ' + self.A.base_type_str
{% else %}
        self.base_type_str = self.A.base_type_str
{% endif %}
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% if type in complex_list %}
        self.base_type_str = 'Conjugate Transposed of ' + self.A.base_type_str
{% else %}
        self.base_type_str = 'Transposed of ' + self.A.base_type_str
{% endif %}
{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]
        self.base_type_str = 'LLSparseMatrixView'
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_common_attributes(self):
        is_OK, attribute = common_matrix_like_attributes(self.C)
        self.assertTrue(is_OK, msg="Attribute '%s' is missing" % attribute)

    def test_nrow_attribute(self):
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
        self.assertTrue(self.C.nrow == self.ncol)
{% else %}
        self.assertTrue(self.C.nrow == self.nrow)
{% endif %}

    def test_ncol_attribute(self):
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
        self.assertTrue(self.C.ncol == self.nrow)
{% else %}
        self.assertTrue(self.C.ncol == self.ncol)
{%  endif %}

    def test_nnz_attribute(self):
        self.assertTrue(self.C.nnz == self.nnz)

    def test_symmetric_storage_attribute(self):
        self.assertTrue(not self.C.store_symmetric)

    def test_zero_storage_attribute(self):
        self.assertTrue(self.C.store_zero)

    def test_is_mutable_attribute(self):
{% if class == 'CSCSparseMatrix' %}
        self.assertTrue(not self.C.is_mutable)
{% elif class == 'CSRSparseMatrix' %}
        self.assertTrue(not self.C.is_mutable)
{% elif class == 'LLSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'TransposedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'ConjugatedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'LLSparseMatrixView' %}
        self.assertTrue(self.C.is_mutable)
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)

    def test_is_symmetric(self):
        self.assertTrue(not self.C.is_symmetric)

##################################
# Case Symmetric, Zero
##################################
class CySparseCommonAttributesSymmetricWithZeroMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 14
        self.nnz = ((self.size + 1) * self.size) / 2
        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetric=True, store_zero=True)
{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
        self.base_type_str = 'CSCSparseMatrix'
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
        self.base_type_str = 'CSRSparseMatrix'
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
        self.base_type_str = 'LLSparseMatrix'
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
        self.base_type_str = 'Transposed of ' + self.A.base_type_str
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% if type in complex_list %}
        self.base_type_str = 'Conjugated of ' + self.A.base_type_str
{% else %}
        self.base_type_str = self.A.base_type_str
{% endif %}
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% if type in complex_list %}
        self.base_type_str = 'Conjugate Transposed of ' + self.A.base_type_str
{% else %}
        self.base_type_str = 'Transposed of ' + self.A.base_type_str
{% endif %}
{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]
        self.base_type_str = 'LLSparseMatrixView'
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

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
{% if class == 'CSCSparseMatrix' %}
        self.assertTrue(not self.C.is_mutable)
{% elif class == 'CSRSparseMatrix' %}
        self.assertTrue(not self.C.is_mutable)
{% elif class == 'LLSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'TransposedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'ConjugatedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.assertTrue(self.C.is_mutable)
{% elif class == 'LLSparseMatrixView' %}
        self.assertTrue(self.C.is_mutable)
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_base_type_str(self):
        self.assertTrue(self.C.base_type_str == self.base_type_str)

    def test_is_symmetric(self):
        self.assertTrue(self.C.is_symmetric)


if __name__ == '__main__':
    unittest.main()