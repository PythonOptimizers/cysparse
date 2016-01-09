#!/usr/bin/env python

"""
This file tests XXX for all matrices objects.

"""

import unittest
from cysparse.sparse.ll_mat import *


########################################################################################################################
# Tests
########################################################################################################################

NROW = 10
NCOL = 14
SIZE = 10


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseUmfpackSolveNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):
        pass

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseUmfpackSolveWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):
        pass

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseUmfpackSolveNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@, store_zero=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):
        pass

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseUmfpackSolveWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True, store_zero=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):
        pass

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


if __name__ == '__main__':
    unittest.main()

