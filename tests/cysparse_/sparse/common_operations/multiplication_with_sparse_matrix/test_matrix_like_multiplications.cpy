#!/usr/bin/env python

"""
This file tests XXX for all matrix-likes objects.

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
class CySparsematrix_like_multiplicationsNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T

{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj

{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):
        pass

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% elif class == 'TransposedSparseMatrix' %}

{% elif class == 'ConjugatedSparseMatrix' %}

{% elif class == 'ConjugateTransposedSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparsematrix_like_multiplicationsWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T

{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj

{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):
        pass

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% elif class == 'TransposedSparseMatrix' %}

{% elif class == 'ConjugatedSparseMatrix' %}

{% elif class == 'ConjugateTransposedSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparsematrix_like_multiplicationsNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T

{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj

{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):
        pass

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% elif class == 'TransposedSparseMatrix' %}

{% elif class == 'ConjugatedSparseMatrix' %}

{% elif class == 'ConjugateTransposedSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparsematrix_like_multiplicationsWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True, store_zero=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T

{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj

{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):
        pass

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% elif class == 'TransposedSparseMatrix' %}

{% elif class == 'ConjugatedSparseMatrix' %}

{% elif class == 'ConjugateTransposedSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


if __name__ == '__main__':
    unittest.main()

