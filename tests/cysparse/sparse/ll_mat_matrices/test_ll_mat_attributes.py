"""
Test ``LLSparseMatrix`` attributes.


"""
from cysparse.types.cysparse_types import *
from cysparse.sparse.ll_mat import *

import unittest


########################################################################################################################
# Tests
########################################################################################################################
class CySparseLLSparseMatrixAttributesBaseTestCase(unittest.TestCase):
    def setUp(self):


        self.nbr_elements = 10
        self.size = 10
        self.nrow = 6
        self.ncol = 6

        self.A = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T, itype=INT32_T)
        self.B = NewLLSparseMatrix(nrow=self.nrow,
                                   ncol=self.ncol,
                                   size_hint=self.nbr_elements,
                                   itype=INT64_T,
                                   dtype=COMPLEX256_T,
                                   store_zeros=True,
                                   is_symmetric=True)

        self.A_T = self.A.T

        self.B_T = self.B.T
        self.B_H = self.B.H
        self.B_conj = self.B.conj


class CySparseLLSparseMatrixAttributesTestCase(CySparseLLSparseMatrixAttributesBaseTestCase):
    """
    Test retrieval of basic attributes.
    """
    def test_basic_attributes(self):

        self.failUnless(self.A.ncol == self.A.nrow == self.size)
        self.failUnless(self.size, self.size == self.A.shape)
        self.failUnless(self.A.nnz == 0)
        self.failUnless(self.A.type == 'LLSparseMatrix')
        self.failUnless(self.A.type_name == 'LLSparseMatrix [INT32_t, FLOAT64_t]')
        self.failUnless(self.A.dtype == FLOAT64_T)
        self.failUnless(self.A.itype == INT32_T)
        self.failUnless(self.A.store_zeros is False)
        self.failUnless(self.A.is_symmetric is False)
        self.failUnless(self.A.is_mutable is True)


        self.failUnless(self.B.nrow == self.nrow)
        self.failUnless(self.B.ncol == self.ncol)
        self.failUnless((self.nrow, self.ncol) == self.B.shape)
        self.failUnless(self.B.nnz == 0)
        self.failUnless(self.B.type == 'LLSparseMatrix')
        self.failUnless(self.B.type_name == 'LLSparseMatrix [INT64_t, COMPLEX256_t]')
        self.failUnless(self.B.dtype == COMPLEX256_T)
        self.failUnless(self.B.itype == INT64_T)
        self.failUnless(self.B.store_zeros is True)
        self.failUnless(self.B.is_symmetric is True)
        self.failUnless(self.B.is_mutable is True)

    def test_moderate_attributes(self):
        """
        Test retrieval of basic attributes of proxies
        """
        # transposed proxy
        self.failUnless(self.A_T.nrow == self.A.ncol)
        self.failUnless(self.A_T.ncol == self.A.nrow)
        self.failUnless((self.A.ncol, self.A.nrow) == self.A_T.shape)
        self.failUnless(self.A.dtype == self.A_T.dtype)
        self.failUnless(self.A.itype == self.A_T.itype)

        # transposed proxy
        self.failUnless(self.B_T.nrow == self.B.ncol)
        self.failUnless(self.B_T.ncol == self.B.nrow)
        self.failUnless((self.B.ncol, self.B.nrow) == self.B_T.shape)
        self.failUnless(self.B.dtype == self.B_T.dtype)
        self.failUnless(self.B.itype == self.B_T.itype)

        # conjugate transposed
        self.failUnless(self.B_H.nrow == self.B.ncol)
        self.failUnless(self.B_H.ncol == self.B.nrow)
        self.failUnless((self.B.ncol, self.B.nrow) == self.B_H.shape)
        self.failUnless(self.B.dtype == self.B_H.dtype)
        self.failUnless(self.B.itype == self.B_H.itype)

        # conjugated
        self.failUnless(self.B_conj.nrow == self.B.nrow)
        self.failUnless(self.B_conj.ncol == self.B.ncol)
        self.failUnless((self.B.nrow, self.B.ncol) == self.B_conj.shape)
        self.failUnless(self.B.dtype == self.B_conj.dtype)
        self.failUnless(self.B.itype == self.B_conj.itype)

    def test_unreachable_attributes(self):
        """
        Test unreachable attributes
        """
        with self.assertRaises(AttributeError):
            print self.A.H  # cannot be reached for real matrix
        with self.assertRaises(AttributeError):
            print self.A.conj  # cannot be reached for real matrix


class CySparseLLSparseMatrixWriteAttributesTestCase(CySparseLLSparseMatrixAttributesBaseTestCase):
    """
    Test writing default (read-only) attributes.
    """
    def test_writing_to_basic_attributes(self):
        with self.assertRaises(AttributeError):
            self.A.nrow = 5
        with self.assertRaises(AttributeError):
            self.A.ncol = 7
        with self.assertRaises(AttributeError):
            self.A.size = 10
        with self.assertRaises(AttributeError):
            self.A.nnz = 45
        with self.assertRaises(AttributeError):
            self.A.type = None
        with self.assertRaises(AttributeError):
            self.A.type_name = 'My type'
        with self.assertRaises(AttributeError):
            self.A.dtype = FLOAT64_T
        with self.assertRaises(AttributeError):
            self.A.itype = INT32_T
        with self.assertRaises(AttributeError):
            self.A.store_zeros = False
        with self.assertRaises(AttributeError):
            self.A.is_symmetric = False
        with self.assertRaises(AttributeError):
            self.A.is_mutable = False

    def test_writing_to_moderate_attributes(self):
        """
        Trying to write to proxies attributes.
        """
        with self.assertRaises(AttributeError):
            self.A_T.nrow = 5
        with self.assertRaises(AttributeError):
            self.A_T.ncol = 7
        with self.assertRaises(AttributeError):
            self.A_T.dtype = FLOAT64_T
        with self.assertRaises(AttributeError):
            self.A_T.itype = INT32_T


if __name__ == '__main__':
    unittest.main()
