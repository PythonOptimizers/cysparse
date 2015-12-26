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
class CySparseCopyNoSymmetryNoZero_CSRSparseMatrix_INT64_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT128_T, itype=INT64_T)


        self.C = self.A.to_csr()


    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """

        self.assertTrue(id(self.C) != id(self.C.copy()))



    def test_copy_element_by_element(self):
        C_copy = self.C.copy()
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.assertTrue(self.C[i, j] == C_copy[i, j])





#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseCopyWithSymmetryNoZero_CSRSparseMatrix_INT64_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=FLOAT128_T, itype=INT64_T, store_symmetry=True)


        self.C = self.A.to_csr()


    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """

        self.assertTrue(id(self.C) != id(self.C.copy()))



    def test_copy_element_by_element(self):
        C_copy = self.C.copy()
        for i in range(self.size):
            for j in range(self.size):
                self.assertTrue(self.C[i, j] == C_copy[i, j])





#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseCopyNoSymmetrySWithZero_CSRSparseMatrix_INT64_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT128_T, itype=INT64_T, store_zero=True)


        self.C = self.A.to_csr()


    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """

        self.assertTrue(id(self.C) != id(self.C.copy()))



    def test_copy_element_by_element(self):
        C_copy = self.C.copy()
        for i in range(self.nrow):
            for j in range(self.ncol):
                self.assertTrue(self.C[i, j] == C_copy[i, j])





#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseCopyWithSymmetrySWithZero_CSRSparseMatrix_INT64_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.size = 10

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=FLOAT128_T, itype=INT64_T, store_symmetry=True, store_zero=True)


        self.C = self.A.to_csr()


    def test_copy_not_same_reference(self):
        """
        Test we have a real deep copy for matrices and views and proxies are singletons.

        Warning:
            If the matrix element type is real, proxies may not be returned.
        """

        self.assertTrue(id(self.C) != id(self.C.copy()))



    def test_copy_element_by_element(self):
        C_copy = self.C.copy()
        for i in range(self.size):
            for j in range(self.size):
                self.assertTrue(self.C[i, j] == C_copy[i, j])





if __name__ == '__main__':
    unittest.main()
