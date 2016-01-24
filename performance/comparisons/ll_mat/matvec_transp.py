"""
This file compares different implementations of ``matvec``, i.e. :math:`A * x`.

We compare the libraries:

- :program:`PySparse`;
- :program:`CySparse` and
- :program:`SciPy.sparse`


"""

import benchmark
import numpy as np

# CySparse
from cysparse.sparse.ll_mat import LLSparseMatrix
from cysparse.common_types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

# PySparse
from pysparse.sparse import spmatrix

# SciPy
from scipy.sparse import lil_matrix


########################################################################################################################
# Helpers
########################################################################################################################
def construct_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3


########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatMatVecTranspBenchmark(benchmark.Benchmark):


    label = "A^t * b with 100 elements and size = 1,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)

    #def tearDown(self):
    #    for i in xrange(self.size):
    #        assert self.w_c[i] == self.w_p[i]
    #        #assert self.w_c[i] == self.w_s[i]


    def test_pysparse(self):
        self.w_p = np.empty(self.size, dtype=np.float64)
        self.A_p.matvec_transp(self.v, self.w_p)
        return

    def test_cysparse(self):
        self.w_c = self.A_c.T * self.v
        return

    def test_cysparse2(self):
        self.w_c = self.A_c.matvec_transp(self.v)
        return


class LLMatMatVecTranspBenchmark_1(LLMatMatVecTranspBenchmark):


    label = "A^t * b with 1,000 elements and size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)


class LLMatMatVecTranspBenchmark_2(LLMatMatVecTranspBenchmark):


    label = "A^t * b with 10,000 elements and size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)

class LLMatMatVecTranspBenchmark_3(LLMatMatVecTranspBenchmark):


    label = "A^t * b with 80,000 elements and size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 80000
        self.size = 100000

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")