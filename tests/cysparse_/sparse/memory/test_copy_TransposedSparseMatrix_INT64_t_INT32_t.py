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
class CySparseCopyNoSymmetryNoZero_TransposedSparseMatrix_INT64_t_INT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT32_T, itype=INT64_T)


        self.C = self.A.T


    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """

         # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()





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


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseCopyWithSymmetryNoZero_TransposedSparseMatrix_INT64_t_INT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=INT32_T, itype=INT64_T, store_symmetry=True)


        self.C = self.A.T


    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """

         # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()





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


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseCopyNoSymmetrySWithZero_TransposedSparseMatrix_INT64_t_INT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=INT32_T, itype=INT64_T, store_zero=True)


        self.C = self.A.T


    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """

         # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()





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


#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseCopyWithSymmetrySWithZero_TransposedSparseMatrix_INT64_t_INT32_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=INT32_T, itype=INT64_T, store_symmetry=True, store_zero=True)


        self.C = self.A.T


    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """

         # proxies **are** singletons
        with self.assertRaises(NotImplementedError):
            self.C.copy()





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


if __name__ == '__main__':
    unittest.main()
