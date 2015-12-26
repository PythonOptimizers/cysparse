#!/usr/bin/env python

"""
This file tests XXX for all sparse-likes objects.

"""

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *


########################################################################################################################
# Tests
########################################################################################################################

NROW = 10
NCOL = 14
SIZE = 10


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseTriangularNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]

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

{% elif class == 'LLSparseMatrixView' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseTriangularWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]

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

{% elif class == 'LLSparseMatrixView' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseTriangularNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]

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

{% elif class == 'LLSparseMatrixView' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseTriangularWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

{% elif class == 'LLSparseMatrixView' %}
        self.C = self.A[:,:]

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

{% elif class == 'LLSparseMatrixView' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


if __name__ == '__main__':
    unittest.main()

