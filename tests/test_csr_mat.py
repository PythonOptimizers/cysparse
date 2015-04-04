from sparse_lib.sparse.csc_mat import CSRSparseMatrix

import unittest


class CSRSparseMatrixSimpleTestCase(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.A = spmatrix.ll_mat(self.n, self.n)
        self.S = spmatrix.ll_mat_sym(self.n)

    def testCreate(self):
        self.failUnless(self.A.shape == (self.n, self.n))
        self.failUnless(self.A.nnz == 0)
        self.failUnless(not self.A.issym)
        self.failUnless(self.S.shape == (self.n, self.n))
        self.failUnless(self.S.nnz == 0)
        self.failUnless(self.S.issym)