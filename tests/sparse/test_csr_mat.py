from sparse_lib.sparse.csr_mat import CSRSparseMatrix

import unittest


class CSRSparseMatrixSimpleTestCase(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.A = CSRSparseMatrix(nrow=self.n, ncol=self.n, nnz=0, is_complex=False)

        print self.A

    def testCreate(self):
        self.failUnless( (self.A.nrow, self.A.ncol) == (self.n, self.n))
        self.failUnless(self.A.nnz == 0)
        self.failUnless(not self.A.is_symmetric)


if __name__ == '__main__':
    unittest.main()

