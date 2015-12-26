#!/usr/bin/env python

"""
This file tests upper and lower triangular sub-matrices for all matrices objects.

"""

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *


########################################################################################################################
# Tests
########################################################################################################################


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseTriangularNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

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

        self.C_tril = self.C.tril()

    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        max_range = min(nrow, ncol)

        for i in range(nrow):
            for j in range(i + 1):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

                if j == max_range:
                    break


    def test_triu_default(self):
        pass


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseTriangularWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

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

        self.C_tril = self.C.tril()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        max_range = min(nrow, ncol)

        for i in range(nrow):
            for j in range(i + 1):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

                if j == max_range:
                    break


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseTriangularNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

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

        self.C_tril = self.C.tril()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        max_range = min(nrow, ncol)

        for i in range(nrow):
            for j in range(i + 1):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

                if j == max_range:
                    break

#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseTriangularWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

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

        self.C_tril = self.C.tril()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        max_range = min(nrow, ncol)

        for i in range(nrow):
            for j in range(i + 1):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

                if j == max_range:
                    break

if __name__ == '__main__':
    unittest.main()

