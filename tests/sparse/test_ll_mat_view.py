from cysparse.sparse.ll_mat import LLSparseMatrix
from cysparse.sparse.ll_mat_view import LLSparseMatrixView

import unittest

import numpy as np


class LLSparseMatrixViewBaseTestCase(unittest.TestCase):
    def setUp(self):
        self.m = 4
        self.n = 3
        self.size_hint = 15
        self.A = LLSparseMatrix(nrow=self.m, ncol=self.n, size_hint=self.size_hint)
        self.A[0, 0] = 89.9
        self.A[1, 0] = -43434.9897
        self.A[3, 2] = -1


########################################################################################################################
# CREATION
########################################################################################################################
class LLSparseMatrixViewCreateTestCase(LLSparseMatrixViewBaseTestCase):
    """
    We test ``LLSparseMatrixView`` **creation**.
    """

    ####################################################################################################################
    # INTEGERS
    ####################################################################################################################
    def testCreateFromIntegersWithSuccess(self):
        """
        A[i, j] shouldn't create any view but only return the corresponding value. All operations here are permitted.
        """
        for i in xrange(self.A.nrow):
            for j in xrange(self.A.ncol):
                self.failUnless(isinstance(self.A[i, j], float))

    def testCreateFromIntegersWithFailure(self):
        """
        A[i, j] shouldn't create any view but only return the corresponding value. None of the operations here
        are permitted.
        """
        with self.assertRaises(IndexError):
            none = self.A[-1, 1]
            none = self.A[-1, -1]
            none = self.A[1, -1]
            none = self.A[0, -1]

    ####################################################################################################################
    # SLICES
    ####################################################################################################################
    def testCreateFromSlicesWithSuccess(self):
        """
        We use both index elements to be slices. All the operations here are permitted.

        """
        # empty slices
        l = slice(0, 0)
        p = slice(0, 0)

        ll_mat_view = self.A[l, p]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failUnless(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == 0)
        self.failUnless(ll_mat_view.ncol == 0)

        # one slice is empty
        ll_mat_view = self.A[0:1, p]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failUnless(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == 1)
        self.failUnless(ll_mat_view.ncol == 0)

        # full slices
        l = slice(0, self.A.nrow)
        p = slice(0, self.A.ncol)
        ll_mat_view = self.A[l, p]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failIf(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == self.A.nrow)
        self.failUnless(ll_mat_view.ncol == self.A.ncol)

        # reverse full slices (this is allowed)
        ll_mat_view = self.A[::-1, ::-1]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failIf(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == self.A.nrow)
        self.failUnless(ll_mat_view.ncol == self.A.ncol)

    ####################################################################################################################
    # LISTS
    ####################################################################################################################
    def testCreateFromListsWithSuccess(self):
        """
        We use both index elements to be lists. All the operations here are permitted.

        """
        # empty lists
        l = []
        p = []
        ll_mat_view = self.A[l, p]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failUnless(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == 0)
        self.failUnless(ll_mat_view.ncol == 0)

        # one list is empty
        ll_mat_view = self.A[[0], p]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failUnless(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == 1)
        self.failUnless(ll_mat_view.ncol == 0)

        # custom list
        ll_mat_view = self.A[[0, 1, 3], [0]]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failIf(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == 3)
        self.failUnless(ll_mat_view.ncol == 1)

    def testCreateFromListsWithFailure(self):
        """
        We use both index elements to be lists. None of the use here is permitted.

        """
        with self.assertRaises(IndexError):
            ll_mat_view = self.A[[0, 1, self.A.nrow], [3]]  # Index out of bound
            ll_mat_view = self.A[[3], [0, -1]] # Index out of bound

        with self.assertRaises(ValueError):
            ll_mat_view = self.A[[1.0], [1]] # elements must be integers!

    ####################################################################################################################
    # NUMPY ARRAYS
    ####################################################################################################################
    def testCreateFromNumpyArraysWithSuccess(self):
        """
        We use both index elements to be lists. All the operations here are permitted.

        """
        # empty numpy arrays
        l = np.array([])
        p = np.array([])
        ll_mat_view = self.A[l, p]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failUnless(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == 0)
        self.failUnless(ll_mat_view.ncol == 0)

        # custom arrays
        ll_mat_view = self.A[np.array([0, 1, 3]), np.array([0])]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failIf(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == 3)
        self.failUnless(ll_mat_view.ncol == 1)

    def testCreateFromNumpyArraysWithFailure(self):
        """
        We use both index elements to be lists. None of the use here is permitted.

        """
        with self.assertRaises(IndexError):
            ll_mat_view = self.A[np.array([0, 1, self.A.nrow], dtype=np.int), np.array([3])]  # Index out of bound
            ll_mat_view = self.A[[3], [0, -1]] # Index out of bound

            ll_mat_view = self.A[np.array([[0,1], [0,2]]), np.array([3])] # wrong dimension of first array

    ####################################################################################################################
    # MIX: EVERYTHING TOGETHER
    ####################################################################################################################
    # this should be a problem as we use the same function for both index elements
    # but we keep one test just in case
    def testCreateFromListAndSliceWithSuccess(self):
        ll_mat_view = self.A[[0,2],0:4:2]
        self.failUnless(isinstance(ll_mat_view, LLSparseMatrixView))
        self.failIf(ll_mat_view.is_empty)
        self.failUnless(ll_mat_view.nrow == 2)
        self.failUnless(ll_mat_view.ncol == 2)


########################################################################################################################
# COPY
########################################################################################################################
class LLSparseMatrixViewCopyTestCase(LLSparseMatrixViewBaseTestCase):
    """
    We test ``LLSparseMatrixView`` ``matrix_copy()`` method. This method returns a corresponding :class:`LLSparseMatrix`.
    """
    def setUp(self):
        super(LLSparseMatrixViewCopyTestCase, self).setUp()

    def testCreateLLSparseMatrixFromEmptyLLSparseMatrixView(self):
        """
        Create an empty :class:`LLSparseMatrix` **with** an empty ``size_hint`` argument which is **not** allowed.
        """
        with self.assertRaises(ValueError):
            ll_mat_empty = self.A[0:0, 0:0].copy()

    def testCreateLLSparseMatrixFromSparseMatrixView(self):
        ll_mat = self.A[0:self.A.nrow, 0:self.A.ncol].copy()
        self.failIf(ll_mat == self.A)
        for i in xrange(self.A.nrow):
            for j in xrange(self.A.ncol):
                self.failUnless(ll_mat[i, j] == self.A[i, j])

########################################################################################################################
# REFERENCES
########################################################################################################################
class LLSparseMatrixViewReferenceTestCase(LLSparseMatrixViewBaseTestCase):
    """
    Test if base matrix is really deleted or not.

    Note:
        I (Nikolaj) did some other tests to check if an :class:`LLSparseMatrix` object really gets deleted or not.
        I don't know how to test the existence of Python objects only using Python.
        A :class:`LLSparseMatrix` object asked to be deleted is indeed deleted whenever all of its views are deleted.
    """
    def setUp(self):
        super(LLSparseMatrixViewReferenceTestCase, self).setUp()

    def testExplicitDeletingOfMatrix(self):
        """
        User deletes explicitly base :class:`LLSparseMatrix`.
        """
        ll_mat_view = self.A[0:2, 0:3:2]

        del self.A

        A = ll_mat_view.get_matrix()
        self.failUnless(A[0, 0] == 89.9)
        ll_mat_view[0, 0] = 2.9
        self.failUnless(A[0, 0] == 2.9)

    def testImplicitDeletingOfMatrix(self):
        """
        Implicit delete.

        """
        def create_view():
            A = LLSparseMatrix(nrow=3, ncol=4, size_hint=10)
            A[0, 0] = 89.9
            A[1, 0] = -43434.9897
            A[2, 2] = -1

            A_view = A[0:2, 0:3:2]

            return A_view

        ll_mat_view = create_view()
        A = ll_mat_view.get_matrix()
        self.failUnless(A[0, 0] == 89.9)
        ll_mat_view[0, 0] = 2.9
        self.failUnless(A[0, 0] == 2.9)


if __name__ == '__main__':
    unittest.main()
