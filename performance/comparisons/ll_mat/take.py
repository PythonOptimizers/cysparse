"""
This file compares different implementations of ``take``.

We compare the libraries:

- :program:`PySparse` and
- :program:`CySparse`.


"""

import benchmark
import numpy as np

# CySparse
from cysparse.sparse.ll_mat import LLSparseMatrix
from cysparse.common_types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

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
class LLMatTakeTripletBenchmark(benchmark.Benchmark):


    label = "Simple take_triplet with 100 elements, size = 1,000 and take_size = 1,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000
        self.take_size = 1000

        assert self.take_size <= self.size

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.arange(0, self.take_size, dtype=np.int32)
        self.id2 = np.full(self.take_size, 37, dtype=np.int32)

        self.b_c = np.empty((self.take_size,),dtype=np.float64)
        self.b_p = np.empty((self.take_size,),dtype=np.float64)

    def tearDown(self):
        for i in xrange(self.take_size):
            assert self.b_c[i] == self.b_p[i]

    def test_pysparse(self):
        self.A_p.take(self.b_p, self.id1, self.id2)
        return

    def test_cysparse(self):
        self.A_c.take_triplet(self.id1, self.id2, self.b_c)
        return

class LLMatTakeTripletBenchmark_1(LLMatTakeTripletBenchmark):


    label = "Simple take_triplet with 1000 elements, size = 10,000 and take_size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000
        self.take_size = 10000

        assert self.take_size <= self.size

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.arange(0, self.take_size, dtype=np.int32)
        self.id2 = np.full(self.take_size, 37, dtype=np.int32)

        self.b_c = np.empty((self.take_size,),dtype=np.float64)
        self.b_p = np.empty((self.take_size,),dtype=np.float64)


class LLMatTakeTripletBenchmark_2(LLMatTakeTripletBenchmark):


    label = "Simple take_triplet with 10000 elements, size = 100,000 and take_size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000
        self.take_size = 100000

        assert self.take_size <= self.size

        self.A_c = LLSparseMatrix(size=self.size, size_hint=self.nbr_elements, itype=INT32_T, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.arange(0, self.take_size, dtype=np.int32)
        self.id2 = np.full(self.take_size, 37, dtype=np.int32)

        self.b_c = np.empty((self.take_size,),dtype=np.float64)
        self.b_p = np.empty((self.take_size,),dtype=np.float64)

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")