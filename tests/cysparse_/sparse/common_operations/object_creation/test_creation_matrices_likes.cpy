#!/usr/bin/env python

"""
This file tests the creation of matrix like objects from :class:`LLSparesMatrix` matrices.

We test **all** types and the symmetric and general cases.
We only use the real parts of complex numbers.

"""
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *
import numpy as np

import unittest

import sys

########################################################################################################################
# Tests
########################################################################################################################
##################################
# Case Non Symmetric, Non Zero
##################################
class CySparseCreationMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)



{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

        print self.A

        print "=" * 80
        print self.C

    def test_equality_one_by_one(self):
        """
        We test matrix-like object equality one element by one element.
        """
{% if class in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix'] %}
        for i in xrange(self.nrow):
            for j in xrange(self.ncol):
                self.assertTrue(self.A[i, j] == self.C[j, i])
{% else %}
        for i in xrange(self.nrow):
            for j in xrange(self.ncol):
                self.assertTrue(self.A[i, j] == self.C[i, j])
{% endif %}


##################################
# Case Symmetric, Non Zero
##################################
class CySparseCreationSymmetricMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 14

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True)

{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_equality_one_by_one(self):
        """
        We test matrix-like object equality one element by one element.
        """
{% if class in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix'] %}
        for i in xrange(self.size):
            for j in xrange(self.size):
                self.assertTrue(self.A[i, j] == self.C[j, i])
{% else %}
        for i in xrange(self.size):
            for j in xrange(self.size):
                self.assertTrue(self.A[i, j] == self.C[i, j])
{% endif %}

##################################
# Case Non Symmetric, Zero
##################################
class CySparseCreationWithZeroMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)

{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_equality_one_by_one(self):
        """
        We test matrix-like object equality one element by one element.
        """
{% if class in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix'] %}
        for i in xrange(self.nrow):
            for j in xrange(self.ncol):
                self.assertTrue(self.A[i, j] == self.C[j, i])
{% else %}
        for i in xrange(self.nrow):
            for j in xrange(self.ncol):
                self.assertTrue(self.A[i, j] == self.C[i, j])
{% endif %}


##################################
# Case Symmetric, Zero
##################################
class CySparseCreationSymmetricWithZeroMatrices_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 14

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True)

{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

    def test_equality_one_by_one(self):
        """
        We test matrix-like object equality one element by one element.
        """
{% if class in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix'] %}
        for i in xrange(self.size):
            for j in xrange(self.size):
                self.assertTrue(self.A[i, j] == self.C[j, i])
{% else %}
        for i in xrange(self.size):
            for j in xrange(self.size):
                self.assertTrue(self.A[i, j] == self.C[i, j])
{% endif %}


if __name__ == '__main__':
    unittest.main()