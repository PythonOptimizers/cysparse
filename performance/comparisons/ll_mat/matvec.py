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
from cysparse.sparse.ll_mat import NewLLSparseMatrix
from cysparse.types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

# PySparse
from pysparse.sparse import spmatrix

# SciPy
from scipy.sparse import lil_matrix

########################################################################################################################
# Helpers
########################################################################################################################
# the same function could be used for all the matrices...

def construct_cysparse_matrix(n, nbr_elements):
    A = NewLLSparseMatrix(size=n, size_hint=nbr_elements, dtype=FLOAT64_T)

    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3

    return A


def construct_pysparse_matrix(n, nbr_elements):
    A = spmatrix.ll_mat(n, n, nbr_elements)

    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3

    return A


def construct_scipy_sparse_matrix(n, nbr_elements):
    A = lil_matrix((n, n), dtype=np.float64)

    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3

    return A


########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatMatVecBenchmark(benchmark.Benchmark):


    label = "matvec with 1000 elements and size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = construct_cysparse_matrix(n=self.size, nbr_elements=self.nbr_elements)
        self.A_p = construct_pysparse_matrix(n=self.size, nbr_elements=self.nbr_elements)
        self.A_s = construct_scipy_sparse_matrix(n=self.size, nbr_elements=self.nbr_elements)



        self.v = np.arange(0, self.size, dtype=np.float64)

    #def tearDown(self):
    #    for i in xrange(self.size):
    #        assert self.w_c[i] == self.w_p[i]
    #        assert self.w_c[i] == self.w_s[i]


    def test_pysparse(self):
        self.w_p = np.empty(self.size, dtype=np.float64)
        self.A_p.matvec(self.v, self.w_p)
        return

    def test_cysparse(self):
        self.w_c = self.A_c * self.v
        return

    def test_cysparse2(self):
        self.A_c.matvec(self.v)
        return

    def test_cysparse3(self):
        self.A_c.matvec2(self.v)

    #def test_scipy_sparse(self):
    #    self.w_s = self.A_s * self.v
    #    return


class LLMatMatVecBenchmark_2(LLMatMatVecBenchmark):


    label = "matvec with 10,000 elements and size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = construct_cysparse_matrix(n=self.size, nbr_elements=self.nbr_elements)
        self.A_p = construct_pysparse_matrix(n=self.size, nbr_elements=self.nbr_elements)
        self.A_s = construct_scipy_sparse_matrix(n=self.size, nbr_elements=self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)


class LLMatMatVecBenchmark_3(LLMatMatVecBenchmark):


    label = "matvec with 100,000 elements and size = 1,000,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100000
        self.size = 1000000

        self.A_c = construct_cysparse_matrix(n=self.size, nbr_elements=self.nbr_elements)
        self.A_p = construct_pysparse_matrix(n=self.size, nbr_elements=self.nbr_elements)
        self.A_s = construct_scipy_sparse_matrix(n=self.size, nbr_elements=self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)


class LLMatMatVecBenchmark_4(LLMatMatVecBenchmark):


    label = "matvec with 500,000 elements and size = 1,000,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 500000
        self.size = 1000000

        self.A_c = construct_cysparse_matrix(n=self.size, nbr_elements=self.nbr_elements)
        self.A_p = construct_pysparse_matrix(n=self.size, nbr_elements=self.nbr_elements)
        self.A_s = construct_scipy_sparse_matrix(n=self.size, nbr_elements=self.nbr_elements)

        self.v = np.arange(0, self.size, dtype=np.float64)

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")