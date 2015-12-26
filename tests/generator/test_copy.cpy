#!/usr/bin/env python

"""
This file tests XXX for all sparse-likes objects.

"""

import unittest


########################################################################################################################
# Tests
########################################################################################################################


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseCopyNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

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

    def test_XXX(self):


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseCopyWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

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

    def test_XXX(self):


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseCopyNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

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

    def test_XXX(self):


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseCopyWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

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

    def test_XXX(self):


if __name__ == '__main__':
    unittest.main()

