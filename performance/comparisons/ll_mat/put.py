"""
This file compares different implementations of ``put``.

We compare the libraries:

- :program:`PySparse`;
- :program:`SPPY` and
- :program:`CySparse`.


"""

import benchmark
import numpy as np

# CySparse
from cysparse.sparse.ll_mat import NewLLSparseMatrix
from cysparse.types.cysparse_types import INT32_T, INT64_T, FLOAT64_T

# PySparse
from pysparse.sparse import spmatrix

# SPPY
from sppy import csarray

########################################################################################################################
# Helpers
########################################################################################################################
def construct_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3


########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatPutTripletBenchmark(benchmark.Benchmark):


    label = "Simple put_triplet with 100 elements, size = 1,000 and put_size = 1,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 100
        self.size = 1000
        self.put_size = 1000

        assert self.put_size <= self.size

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.A_sppy = None

        self.id1 = np.arange(0, self.put_size, dtype=np.int32)
        self.id2 = np.full(self.put_size, 37, dtype=np.int32)

        self.b = np.arange(0, self.put_size,dtype=np.float64)


    def eachSetUp(self):
        self.A_sppy = csarray((self.size, self.size), dtype=np.float64, storagetype='row')

    #def tearDown(self):
    #    for i in xrange(self.size):
    #        for j in xrange(self.size):
    #            assert self.A_c[i, j] == self.A_p[i, j]

    def test_pysparse(self):
        self.A_p.put(self.b, self.id1, self.id2)
        return

    def test_cysparse(self):
        self.A_c.put_triplet(self.id1, self.id2, self.b)
        return

    def test_sppy(self):
        self.A_sppy.put(self.b, self.id1, self.id2, init=True)


class LLMatPutTripletBenchmark_1(LLMatPutTripletBenchmark):


    label = "Simple put_triplet with 1,000 elements, size = 10,000 and put_size = 10,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 1000
        self.size = 10000
        self.put_size = 10000

        assert self.put_size <= self.size

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.arange(0, self.put_size, dtype=np.int32)
        self.id2 = np.full(self.put_size, 37, dtype=np.int32)

        self.b = np.arange(0, self.put_size,dtype=np.float64)

class LLMatPutTripletBenchmark_2(LLMatPutTripletBenchmark):


    label = "Simple put_triplet with 10,000 elements, size = 100,000 and put_size = 100,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 100000
        self.put_size = 100000

        assert self.put_size <= self.size

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.arange(0, self.put_size, dtype=np.int32)
        self.id2 = np.full(self.put_size, 37, dtype=np.int32)

        self.b = np.arange(0, self.put_size,dtype=np.float64)


class LLMatPutTripletBenchmark_3(LLMatPutTripletBenchmark):


    label = "Simple put_triplet with 10,000 elements, size = 1,000,000 and put_size = 800,000"
    each = 100


    def setUp(self):

        self.nbr_elements = 10000
        self.size = 1000000
        self.put_size = 800000

        assert self.put_size <= self.size

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.id1 = np.arange(0, self.put_size, dtype=np.int32)
        self.id2 = np.full(self.put_size, 37, dtype=np.int32)

        self.b = np.arange(0, self.put_size,dtype=np.float64)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")