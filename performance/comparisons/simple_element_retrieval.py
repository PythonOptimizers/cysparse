"""
This file compares different implementations of simple element retrieval, i.e. :math:`A[i, j]`.

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
def construct_sparse_matrix(A, n, nbr_elements):
    for i in xrange(nbr_elements):
        A[i % n, (2 * i + 1) % n] = i / 3


def retrieve_all_elements(A, n, x):
    for i in xrange(n):
        for j in xrange(n):
            x.append(A[i, j])

########################################################################################################################
# Benchmark
########################################################################################################################
class LLMatSimpleElementRetrievalBenchmark(benchmark.Benchmark):


    label = "Simple element retrieval with 1000 elements and size = 10,000"
    each = 10


    def setUp(self):

        self.nbr_elements = 10
        self.size = 100

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)
        self.x_c = []

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)
        self.x_p = []

        #self.A_s = lil_matrix((self.size, self.size), dtype=np.float64) # how do we reserve space in advance?
        #construct_sparse_matrix(self.A_s, self.size, self.nbr_elements)

    def tearDown(self):
        size = len(self.x_c)

        assert len(self.x_p) == size
        for i in xrange(size):
            assert self.x_c[i] == self.x_p[i]

    def test_pysparse(self):
        retrieve_all_elements(self.A_p, self.size, self.x_p)
        return

    def test_cysparse(self):
        retrieve_all_elements(self.A_c, self.size, self.x_c)
        return

    #def test_scipy_sparse(self):
    #    retrieve_all_elements(self.A_s, self.size)
    #    return

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")