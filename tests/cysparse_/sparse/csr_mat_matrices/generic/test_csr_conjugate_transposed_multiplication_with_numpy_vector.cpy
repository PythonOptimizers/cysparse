#!/usr/bin/env python

"""
This file tests the **conjugate transposed** multiplication of an :class:`CSRSparseMatrix` matrix with a :program:`NumPy` vector.

We test **all** **complex** types and the symmetric and general cases. We also test strided vectors.


     If this is a python script (.py), it has been automatically generated by the 'generate_code.py' script.

"""
from cysparse.sparse.ll_mat import *
from cysparse.types.cysparse_types import *
import numpy as np

import unittest

import sys

########################################################################################################################
# Tests
########################################################################################################################
class CySparseCSRConjugateTransposedMultiplicationWithANumpyVectorBaseTestCase(unittest.TestCase):
    def setUp(self):
        pass

class CySparseCSRConjugateTransposedMultiplicationWithANumpyVectorTestCase(CySparseCSRConjugateTransposedMultiplicationWithANumpyVectorBaseTestCase):
    """
    Basic case: ``y = A^h * x`` with ``A`` **non** symmetric and ``x`` and ``y`` without strides.
    """
    def setUp(self):
        self.nbr_of_elements = 10
        self.nrow = 4
        self.ncol = 6

{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in complex_list %}
        #self.l_@outerloop.index@_@loop.index@ = NewLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, size_hint=self.nbr_of_elements, itype=@index_type|type2enum@, dtype=@element_type|type2enum@)
        #construct_sparse_matrix(self.l_@outerloop.index@_@loop.index@, self.nrow, self.ncol, self.nbr_of_elements)

        self.l_@outerloop.index@_@loop.index@ = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=@index_type|type2enum@, dtype=@element_type|type2enum@, row_wise=False)

        self.l_@outerloop.index@_@loop.index@_csr = self.l_@outerloop.index@_@loop.index@.to_csr()
  {% endfor %}
{% endfor %}

{% for element_type in complex_list %}
        self.x_@element_type@ = np.empty(self.nrow, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_@element_type@.fill(1+2j)
{% endfor %}

    def test_simple_multiplication_one_by_one(self):
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in complex_list %}
        l_y = self.l_@outerloop.index@_@loop.index@.matvec_htransp(self.x_@element_type@)
        csr_y = self.l_@outerloop.index@_@loop.index@_csr.matvec_htransp(self.x_@element_type@)
        for i in xrange(self.ncol):
            self.failUnless(l_y[i] == csr_y[i])
  {% endfor %}
{% endfor %}


class CySparseSymCSRConjugateTransposedMultiplicationWithANumpyVectorTestCase(CySparseCSRConjugateTransposedMultiplicationWithANumpyVectorBaseTestCase):
    """
    Basic case: ``y = A^h * x`` with ``A`` symmetric and ``x`` and ``y`` without strides.
    """
    def setUp(self):
        self.nbr_of_elements = 10
        self.size = 100
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in complex_list %}
        #self.l_@outerloop.index@_@loop.index@ = NewLLSparseMatrix(__is_symmetric=True, size=self.size, size_hint=self.nbr_of_elements, itype=@index_type|type2enum@, dtype=@element_type|type2enum@)
        #construct_sym_sparse_matrix(self.l_@outerloop.index@_@loop.index@, self.size, self.nbr_of_elements)

        self.l_@outerloop.index@_@loop.index@ = NewLinearFillLLSparseMatrix(size=self.size, itype=@index_type|type2enum@, dtype=@element_type|type2enum@, row_wise=False, is_symmetric=True)

        self.l_@outerloop.index@_@loop.index@_csr = self.l_@outerloop.index@_@loop.index@.to_csr()
  {% endfor %}
{% endfor %}

{% for element_type in complex_list %}
        self.x_@element_type@ = np.empty(self.size, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_@element_type@.fill(1+2j)
{% endfor %}

    def test_simple_multiplication_one_by_one(self):
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in complex_list %}
        l_y = self.l_@outerloop.index@_@loop.index@.matvec_htransp(self.x_@element_type@)
        csr_y = self.l_@outerloop.index@_@loop.index@_csr.matvec_htransp(self.x_@element_type@)
        for i in xrange(self.size):
            self.failUnless(l_y[i] == csr_y[i])
  {% endfor %}
{% endfor %}


class CySparseCSRConjugateTransposedMultiplicationWithAStridedNumpyVectorTestCase(CySparseCSRConjugateTransposedMultiplicationWithANumpyVectorBaseTestCase):
    """
    Basic case: ``y = A^h * x`` with ``A`` **non** symmetric and ``x`` **with** strides.
    """
    def setUp(self):
        self.nbr_of_elements = 10
        self.nrow = 80
        self.ncol = 100

        self.stride_factor = 10

{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in complex_list %}
        #self.l_@outerloop.index@_@loop.index@ = NewLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, size_hint=self.nbr_of_elements, itype=@index_type|type2enum@, dtype=@element_type|type2enum@)
        #construct_sparse_matrix(self.l_@outerloop.index@_@loop.index@, self.nrow, self.ncol, self.nbr_of_elements)

        self.l_@outerloop.index@_@loop.index@ = NewLinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, itype=@index_type|type2enum@, dtype=@element_type|type2enum@, row_wise=False)

        self.l_@outerloop.index@_@loop.index@_csr = self.l_@outerloop.index@_@loop.index@.to_csr()
  {% endfor %}
{% endfor %}

{% for element_type in complex_list %}
        self.x_@element_type@ = np.empty(self.nrow, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_@element_type@.fill(1+2j)

        self.x_strided_@element_type@ = np.empty(self.nrow * self.stride_factor, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_strided_@element_type@.fill(2)

        for i in xrange(self.nrow):
            self.x_strided_@element_type@[i * self.stride_factor] = self.x_@element_type@[i]
{% endfor %}

    def test_simple_multiplication_one_by_one(self):
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in complex_list %}
        l_y = self.l_@outerloop.index@_@loop.index@.matvec_htransp(self.x_@element_type@)
        csr_y = self.l_@outerloop.index@_@loop.index@_csr.matvec_htransp(self.x_strided_@element_type@[::self.stride_factor])
        for i in xrange(self.ncol):
            self.failUnless(l_y[i] == csr_y[i])
  {% endfor %}
{% endfor %}


class CySparseSymCSRConjugateTransposedMultiplicationWithAStridedNumpyVectorTestCase(CySparseCSRConjugateTransposedMultiplicationWithANumpyVectorBaseTestCase):
    """
    Basic case: ``y = A^h * x`` with ``A`` **symmetric** and ``x`` **with** strides.
    """
    def setUp(self):
        self.nbr_of_elements = 10
        self.size = 100

        self.stride_factor = 10

{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in complex_list %}
        #self.l_@outerloop.index@_@loop.index@ = NewLLSparseMatrix(__is_symmetric=True, size=self.size, size_hint=self.nbr_of_elements, itype=@index_type|type2enum@, dtype=@element_type|type2enum@)
        #construct_sym_sparse_matrix(self.l_@outerloop.index@_@loop.index@, self.size, self.nbr_of_elements)

        self.l_@outerloop.index@_@loop.index@ = NewLinearFillLLSparseMatrix(size=self.size, itype=@index_type|type2enum@, dtype=@element_type|type2enum@, row_wise=False, is_symmetric=True)

        self.l_@outerloop.index@_@loop.index@_csr = self.l_@outerloop.index@_@loop.index@.to_csr()
  {% endfor %}
{% endfor %}

{% for element_type in complex_list %}
        self.x_@element_type@ = np.empty(self.size, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_@element_type@.fill(1+2j)

        self.x_strided_@element_type@ = np.empty(self.size * self.stride_factor, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_strided_@element_type@.fill(2)

        for i in xrange(self.size):
            self.x_strided_@element_type@[i * self.stride_factor] = self.x_@element_type@[i]
{% endfor %}

    def test_simple_multiplication_one_by_one(self):
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in complex_list %}
        l_y = self.l_@outerloop.index@_@loop.index@.matvec_htransp(self.x_@element_type@)
        csr_y = self.l_@outerloop.index@_@loop.index@_csr.matvec_htransp(self.x_strided_@element_type@[::self.stride_factor])
        for i in xrange(self.size):
            self.failUnless(l_y[i] == csr_y[i])
  {% endfor %}
{% endfor %}

if __name__ == '__main__':
    unittest.main()