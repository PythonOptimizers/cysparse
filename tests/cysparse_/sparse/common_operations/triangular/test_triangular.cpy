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
{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

        self.C_tril = self.C.tril()
        self.C_triu = self.C.triu()

    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for i in range(nrow):
            for j in range(min(i + 1, ncol)):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

    def test_tril_general(self):
        """
        Test ``tril(k)`` with ``k < 0``.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(-nrow, 0, 1):

            for i in range(nrow):
                for j in range(min(i + k + 1, ncol)):
                    self.assertTrue(self.C_tril[i, j] == self.A[i, j])

    def test_triu_default(self):
        """
        Test ``triu()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for j in range(ncol):
            for i in range(min(nrow, j)):
                self.assertTrue(self.C_triu[i, j] == self.A[i, j])

    def test_triu_general(self):
        """
        Test ``triu(k)`` with ``k > 0``.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(1, ncol, 1):
            for j in range(ncol):
                for i in range(min(nrow, j - k + 1)):
                    self.assertTrue(self.C_triu[i, j] == self.A[i, j])

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
{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

        self.C_tril = self.C.tril()
        self.C_triu = self.C.triu()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for i in range(nrow):
            for j in range(min(i + 1, ncol)):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

    def test_tril_general(self):
        """
        Test ``tril(k)`` with ``k < 0``.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(-nrow, 0, 1):

            for i in range(nrow):
                for j in range(min(i + k + 1, ncol)):
                    self.assertTrue(self.C_tril[i, j] == self.A[i, j])

    def test_triu_default(self):
        """
        Test ``triu()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for j in range(ncol):
            for i in range(min(nrow, j)):
                self.assertTrue(self.C_triu[i, j] == self.A[i, j])

    def test_triu_general(self):
        """
        Test ``triu(k)`` with ``k > 0``.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(1, ncol, 1):
            for j in range(ncol):
                for i in range(min(nrow, j - k + 1)):
                    self.assertTrue(self.C_triu[i, j] == self.A[i, j])

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
{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

        self.C_tril = self.C.tril()
        self.C_triu = self.C.triu()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for i in range(nrow):
            for j in range(min(i + 1, ncol)):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

    def test_tril_general(self):
        """
        Test ``tril(k)`` with ``k < 0``.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(-nrow, 0, 1):

            for i in range(nrow):
                for j in range(min(i + k + 1, ncol)):
                    self.assertTrue(self.C_tril[i, j] == self.A[i, j])

    def test_triu_default(self):
        """
        Test ``triu()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for j in range(ncol):
            for i in range(min(nrow, j)):
                self.assertTrue(self.C_triu[i, j] == self.A[i, j])

    def test_triu_general(self):
        """
        Test ``triu(k)`` with ``k > 0``.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(1, ncol, 1):
            for j in range(ncol):
                for i in range(min(nrow, j - k + 1)):
                    self.assertTrue(self.C_triu[i, j] == self.A[i, j])

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
{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

        self.C_tril = self.C.tril()
        self.C_triu = self.C.triu()


    def test_tril_default(self):
        """
        Test ``tril()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for i in range(nrow):
            for j in range(min(i + 1, ncol)):
                self.assertTrue(self.C_tril[i, j] == self.A[i, j])

    def test_tril_general(self):
        """
        Test ``tril(k)`` with ``k < 0``.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(-nrow, 0, 1):

            for i in range(nrow):
                for j in range(min(i + k + 1, ncol)):
                    self.assertTrue(self.C_tril[i, j] == self.A[i, j])

    def test_triu_default(self):
        """
        Test ``triu()`` with default arguments.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for j in range(ncol):
            for i in range(min(nrow, j)):
                self.assertTrue(self.C_triu[i, j] == self.A[i, j])

    def test_triu_general(self):
        """
        Test ``triu(k)`` with ``k > 0``.
        """
        nrow = self.C.nrow
        ncol = self.C.ncol

        for k in range(1, ncol, 1):
            for j in range(ncol):
                for i in range(min(nrow, j - k + 1)):
                    self.assertTrue(self.C_triu[i, j] == self.A[i, j])

if __name__ == '__main__':
    unittest.main()

