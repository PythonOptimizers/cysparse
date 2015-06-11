#!/usr/bin/env python

"""
This file tests the multiplication of an :class:`CSRSparseMatrix` matricix with a :program:`NumPy` vector.

We test **all** types and the symmetric and general cases. We also test strided vectors.
We only use the real parts of complex numbers.


     If this is a python script (.py), it has been automatically generated by the 'generate_code.py' script.

"""
from cysparse.sparse.ll_mat import *
from cysparse.types.cysparse_types import *
import numpy as np

import unittest

import sys

########################################################################################################################
# Helpers
########################################################################################################################
def construct_sym_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        k = i % n
        p = (i % 2 + 1) % n
        if k >= p:
            A[k, p] = i / 3
        else:
            A[p, k] = i / 3

def construct_sparse_matrix(A, m, n, nbr_elements):
    for i in xrange(nbr_elements):
        k = i % n
        p = (i % 2 + 1) % m
        A[p, k] = i / 3

########################################################################################################################
# Tests
########################################################################################################################
class CySparseCSRMultiplicationWithANumpyVectorBaseTestCase(unittest.TestCase):
    def setUp(self):
        pass

class CySparseCSRMultiplicationWithANumpyVectorTestCase(CySparseCSRMultiplicationWithANumpyVectorBaseTestCase):
    """
    Basic case: ``y = A * x`` with ``A`` **non** symmetric and ``x`` and ``y`` without strides.
    """
    def setUp(self):
        self.nbr_of_elements = 10
        self.nrow = 80
        self.ncol = 100

{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in type_list %}
        self.l_@outerloop.index@_@loop.index@ = NewLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, size_hint=self.nbr_of_elements, itype=@index_type|type2enum@, dtype=@element_type|type2enum@)
        construct_sparse_matrix(self.l_@outerloop.index@_@loop.index@, self.nrow, self.ncol, self.nbr_of_elements)

        self.l_@outerloop.index@_@loop.index@_csr = self.l_@outerloop.index@_@loop.index@.to_csr()
  {% endfor %}
{% endfor %}

{% for element_type in type_list %}
        self.x_@element_type@ = np.ones(self.ncol, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
{% endfor %}

    def test_simple_multiplication_one_by_one(self):
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in type_list %}
        l_y = self.l_@outerloop.index@_@loop.index@.matvec(self.x_@element_type@)
        csr_y = self.l_@outerloop.index@_@loop.index@_csr.matvec(self.x_@element_type@)
        for i in xrange(self.nrow):
            self.failUnless(l_y[i] == csr_y[i])
  {% endfor %}
{% endfor %}


class CySparseSymCSRMultiplicationWithANumpyVectorTestCase(CySparseCSRMultiplicationWithANumpyVectorBaseTestCase):
    """
    Basic case: ``y = A * x`` with ``A`` symmetric and ``x`` and ``y`` without strides.
    """
    def setUp(self):
        self.nbr_of_elements = 10
        self.size = 100
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in type_list %}
        self.l_@outerloop.index@_@loop.index@ = NewLLSparseMatrix(is_symmetric=True, size=self.size, size_hint=self.nbr_of_elements, itype=@index_type|type2enum@, dtype=@element_type|type2enum@)
        construct_sym_sparse_matrix(self.l_@outerloop.index@_@loop.index@, self.size, self.nbr_of_elements)

        self.l_@outerloop.index@_@loop.index@_csr = self.l_@outerloop.index@_@loop.index@.to_csr()
  {% endfor %}
{% endfor %}

{% for element_type in type_list %}
        self.x_@element_type@ = np.ones(self.size, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
{% endfor %}

    def test_simple_multiplication_one_by_one(self):
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in type_list %}
        l_y = self.l_@outerloop.index@_@loop.index@.matvec(self.x_@element_type@)
        csr_y = self.l_@outerloop.index@_@loop.index@_csr.matvec(self.x_@element_type@)
        for i in xrange(self.size):
            self.failUnless(l_y[i] == csr_y[i])
  {% endfor %}
{% endfor %}


class CySparseCSRMultiplicationWithAStridedNumpyVectorTestCase(CySparseCSRMultiplicationWithANumpyVectorBaseTestCase):
    """
    Basic case: ``y = A * x`` with ``A`` **symmetric** and ``x`` **with** strides.
    """
    def setUp(self):
        self.nbr_of_elements = 10
        self.size = 100

        self.stride_factor = 10

{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in type_list %}
        self.l_@outerloop.index@_@loop.index@ = NewLLSparseMatrix(is_symmetric=True, size=self.size, size_hint=self.nbr_of_elements, itype=@index_type|type2enum@, dtype=@element_type|type2enum@)
        construct_sym_sparse_matrix(self.l_@outerloop.index@_@loop.index@, self.size, self.nbr_of_elements)

        self.l_@outerloop.index@_@loop.index@_csr = self.l_@outerloop.index@_@loop.index@.to_csr()
  {% endfor %}
{% endfor %}

{% for element_type in type_list %}
        self.x_@element_type@ = np.ones(self.size, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_strided_@element_type@ = np.empty(self.size * self.stride_factor, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_strided_@element_type@.fill(2)

        for i in xrange(self.size):
            self.x_strided_@element_type@[i * self.stride_factor] = self.x_@element_type@[i]
{% endfor %}

    def test_simple_multiplication_one_by_one(self):
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in type_list %}
        l_y = self.l_@outerloop.index@_@loop.index@.matvec(self.x_@element_type@)
        csr_y = self.l_@outerloop.index@_@loop.index@_csr.matvec(self.x_strided_@element_type@[::self.stride_factor])
        for i in xrange(self.size):
            self.failUnless(l_y[i] == csr_y[i])
  {% endfor %}
{% endfor %}


class CySparseSymCSRMultiplicationWithAStridedNumpyVectorTestCase(CySparseCSRMultiplicationWithANumpyVectorBaseTestCase):
    """
    Basic case: ``y = A * x`` with ``A`` **symmetric** and ``x`` **with** strides.
    """
    def setUp(self):
        self.nbr_of_elements = 10
        self.size = 100

        self.stride_factor = 10

{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in type_list %}
        self.l_@outerloop.index@_@loop.index@ = NewLLSparseMatrix(is_symmetric=True, size=self.size, size_hint=self.nbr_of_elements, itype=@index_type|type2enum@, dtype=@element_type|type2enum@)
        construct_sym_sparse_matrix(self.l_@outerloop.index@_@loop.index@, self.size, self.nbr_of_elements)

        self.l_@outerloop.index@_@loop.index@_csr = self.l_@outerloop.index@_@loop.index@.to_csr()
  {% endfor %}
{% endfor %}

{% for element_type in type_list %}
        self.x_@element_type@ = np.ones(self.size, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_strided_@element_type@ = np.empty(self.size * self.stride_factor, dtype=np.@element_type|type2enum|cysparse_type_to_numpy_type@)
        self.x_strided_@element_type@.fill(2)

        for i in xrange(self.size):
            self.x_strided_@element_type@[i * self.stride_factor] = self.x_@element_type@[i]
{% endfor %}

    def test_simple_multiplication_one_by_one(self):
{% for index_type in index_list %}
  {% set outerloop = loop %}
  {% for element_type in type_list %}
        l_y = self.l_@outerloop.index@_@loop.index@.matvec(self.x_@element_type@)
        csr_y = self.l_@outerloop.index@_@loop.index@_csr.matvec(self.x_strided_@element_type@[::self.stride_factor])
        for i in xrange(self.size):
            self.failUnless(l_y[i] == csr_y[i])
  {% endfor %}
{% endfor %}


if __name__ == '__main__':
    unittest.main()