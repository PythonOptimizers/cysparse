#!/usr/bin/env python

"""
This file tests ``find()`` for all matrices objects.

"""

import unittest
import numpy as np
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
class CySparseFindNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

    def test_find_element_by_element(self):
        """
        Test ``find()``.

        """
        C_i, C_j, C_v = self.C.find()

        for k in range(self.A.nnz):
            self.assertTrue(self.C[C_i[k], C_j[k]] == C_v[k])




#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseFindWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

    def test_find_element_by_element(self):
        """
        Test ``find()``.

        """
        C_i, C_j, C_v = self.C.find()

        for k in range(self.A.nnz):
            self.assertTrue(self.C[C_i[k], C_j[k]] == C_v[k])


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseFindNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

    def test_find_element_by_element(self):
        """
        Test ``find()``.

        """
        C_i, C_j, C_v = self.C.find()

        for k in range(self.A.nnz):
            self.assertTrue(self.C[C_i[k], C_j[k]] == C_v[k])


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseFindWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

    def test_find_element_by_element(self):
        """
        Test ``find()``.

        """
        C_i, C_j, C_v = self.C.find()

        for k in range(self.A.nnz):
            self.assertTrue(self.C[C_i[k], C_j[k]] == C_v[k])



if __name__ == '__main__':
    unittest.main()

