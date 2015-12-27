#!/usr/bin/env python

"""
This file tests ``copy()`` for all sparse-likes objects.

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
class CySparseCopyNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
         # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()
{% elif class in ['ConjugatedSparseMatrix'] and type in complex_list %}
        # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()
{% else %}
        self.assertTrue(id(self.C) != id(self.C.copy()))
{% endif %}

{% if class not in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix', 'ConjugatedSparseMatrix'] %}
    def test_copy_element_by_element(self):
        C_copy = self.C.copy()
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.assertTrue(self.C[i, j] == C_copy[i, j])

{% endif %}

{% if class in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix', 'LLSparseMatrixView'] or (class == 'ConjugatedSparseMatrix' and type in complex_list) %}
    def test_matrix_copy_not_same_reference(self):
        """
        Test ``matrix_copy()`` doesn't return same reference as initial object.
        """
        C_matrix_copy = self.C.matrix_copy()
        self.assertTrue(id(self.C) != id(C_matrix_copy))

    def test_matrix_copy_element_by_element(self):
        C_matrix_copy = self.C.matrix_copy()
        nrow = C_matrix_copy.nrow
        ncol = C_matrix_copy.ncol

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i, j] == C_matrix_copy[i, j])
{% endif %}

#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseCopyWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

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

    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
         # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()
{% elif class in ['ConjugatedSparseMatrix'] and type in complex_list %}
        # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()
{% else %}
        self.assertTrue(id(self.C) != id(self.C.copy()))
{% endif %}

{% if class not in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix', 'ConjugatedSparseMatrix'] %}
    def test_copy_element_by_element(self):
        C_copy = self.C.copy()
        for i in range(self.size):
            for j in range(self.size):
                self.assertTrue(self.C[i, j] == C_copy[i, j])

{% endif %}

{% if class in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix', 'LLSparseMatrixView'] or (class == 'ConjugatedSparseMatrix' and type in complex_list) %}
    def test_matrix_copy_not_same_reference(self):
        """
        Test ``matrix_copy()`` doesn't return same reference as initial object.
        """
        C_matrix_copy = self.C.matrix_copy()
        self.assertTrue(id(self.C) != id(C_matrix_copy))

    def test_matrix_copy_element_by_element(self):
        C_matrix_copy = self.C.matrix_copy()
        nrow = C_matrix_copy.nrow
        ncol = C_matrix_copy.ncol

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i, j] == C_matrix_copy[i, j])
{% endif %}

#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseCopyNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
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

    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
         # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()
{% elif class in ['ConjugatedSparseMatrix'] and type in complex_list %}
        # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()
{% else %}
        self.assertTrue(id(self.C) != id(self.C.copy()))
{% endif %}

{% if class not in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix', 'ConjugatedSparseMatrix'] %}
    def test_copy_element_by_element(self):
        C_copy = self.C.copy()
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.assertTrue(self.C[i, j] == C_copy[i, j])

{% endif %}

{% if class in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix', 'LLSparseMatrixView'] or (class == 'ConjugatedSparseMatrix' and type in complex_list) %}
    def test_matrix_copy_not_same_reference(self):
        """
        Test ``matrix_copy()`` doesn't return same reference as initial object.
        """
        C_matrix_copy = self.C.matrix_copy()
        self.assertTrue(id(self.C) != id(C_matrix_copy))

    def test_matrix_copy_element_by_element(self):
        C_matrix_copy = self.C.matrix_copy()
        nrow = C_matrix_copy.nrow
        ncol = C_matrix_copy.ncol

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i, j] == C_matrix_copy[i, j])
{% endif %}

#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseCopyWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

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

    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
         # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()
{% elif class in ['ConjugatedSparseMatrix'] and type in complex_list %}
        # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()
{% else %}
        self.assertTrue(id(self.C) != id(self.C.copy()))
{% endif %}

{% if class not in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix', 'ConjugatedSparseMatrix'] %}
    def test_copy_element_by_element(self):
        C_copy = self.C.copy()
        for i in range(self.size):
            for j in range(self.size):
                self.assertTrue(self.C[i, j] == C_copy[i, j])

{% endif %}

{% if class in ['TransposedSparseMatrix', 'ConjugateTransposedSparseMatrix', 'LLSparseMatrixView'] or (class == 'ConjugatedSparseMatrix' and type in complex_list) %}
    def test_matrix_copy_not_same_reference(self):
        """
        Test ``matrix_copy()`` doesn't return same reference as initial object.
        """
        C_matrix_copy = self.C.matrix_copy()
        self.assertTrue(id(self.C) != id(C_matrix_copy))

    def test_matrix_copy_element_by_element(self):
        C_matrix_copy = self.C.matrix_copy()
        nrow = C_matrix_copy.nrow
        ncol = C_matrix_copy.ncol

        for i in range(nrow):
            for j in range(ncol):
                self.assertTrue(self.C[i, j] == C_matrix_copy[i, j])
{% endif %}

if __name__ == '__main__':
    unittest.main()

