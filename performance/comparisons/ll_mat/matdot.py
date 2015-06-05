"""
This file compares different implementations of `matdot`, i.e. :math:`A^t * B`.

We compare the libraries:

- :program:`PySparse` and
- :program:`CySparse`.


"""

import benchmark
import numpy as np

# CySparse
from cysparse.sparse.ll_mat import NewLLSparseMatrix
from cysparse.types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

# PySparse
from pysparse.sparse import spmatrix


########################################################################################################################
# Helpers
########################################################################################################################
def construct_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3

########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatMatDotBenchmark(benchmark.Benchmark):


    label = "Simple matdot (A^t * B) with 100 elements and size = 1,000"
    each = 10


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.C_c = None
        self.C_c_via_T = None
        self.C_p = None

    #def tearDown(self):
    #    for i in xrange(self.size):
    #        for j in xrange(self.size):
    #            assert self.C_c[i,j] == self.C_p[i,j]

    def test_pysparse(self):
        self.C_p = spmatrix.dot(self.A_p, self.A_p)
        return

    def test_cysparse(self):
        self.C_c = self.A_c.matdot_transp(self.A_c)
        return

    def test_cysparse2(self):
        self.C_c = self.A_c.T * self.A_c
        return


class LLMatMatDotBenchmark_1(LLMatMatDotBenchmark):


    label = "Simple matdot (A^t * B) with 1,000 elements and size = 10,000"
    each = 10


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.C_c = None
        self.C_c_via_T = None
        self.C_p = None


class LLMatMatDotBenchmark_2(LLMatMatDotBenchmark):


    label = "Simple matdot (A^t * B) with 10,000 elements and size = 100,000"
    each = 10


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.C_c = None
        self.C_c_via_T = None
        self.C_p = None


class LLMatMatDotBenchmark_3(LLMatMatDotBenchmark):


    label = "Simple matdot (A^t * B) with 80,000 elements and size = 100,000"
    each = 10


    def setUp(self):

        self.nbr_elements = 80000
        self.size = 100000

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.C_c = None
        self.C_c_via_T = None
        self.C_p = None

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")