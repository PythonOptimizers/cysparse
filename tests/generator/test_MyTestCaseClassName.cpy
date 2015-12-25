#!/usr/bin/env python

"""
This file tests XXX for all matrices objects.

"""

import unittest


########################################################################################################################
# Tests
########################################################################################################################


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseMyTestCaseClassNameNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseMyTestCaseClassNameWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseMyTestCaseClassNameNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseMyTestCaseClassNameWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

{% if class == 'LLSparseMatrix' %}

{% elif class == 'CSCSparseMatrix' %}

{% elif class == 'CSRSparseMatrix' %}

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_XXX(self):


if __name__ == '__main__':
    unittest.main()

