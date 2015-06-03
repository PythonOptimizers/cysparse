"""
This file compares different implementations of ``delete_row_with_mask``.

We compare the libraries:

- :program:`PySparse` and
- :program:`CySparse`.


"""

import benchmark
import numpy as np
import sys
import time

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
class LLMatDeleteRowsBenchmark(benchmark.Benchmark):


    label = "Simple delete_rows with 100 elements, size = 1,000 and 100 row to delete"
    each = 2



    def eachSetUp(self):

        print "tentative eachSetup"



        print "eachSetup"



        self.nbr_elements = 100
        self.size = 10
        self.nbr_of_rows_to_delete = 9

        assert self.size > self.nbr_of_rows_to_delete

        self.A_c = NewLLSparseMatrix(size=self.size, size_hint=self.nbr_elements, dtype=FLOAT64_T)
        construct_sparse_matrix(self.A_c, self.size, self.nbr_elements)

        self.A_p = spmatrix.ll_mat(self.size, self.size, self.nbr_elements)
        construct_sparse_matrix(self.A_p, self.size, self.nbr_elements)

        self.mask_c = np.ones(self.size, dtype=np.int8)
        self.mask_p = np.ones(self.size, 'l')


        self.mask_c[0:self.nbr_of_rows_to_delete] = 0
        self.mask_p[0:self.nbr_of_rows_to_delete] = 0

        print "end each setup"


    def tearDown(self):
        time.sleep(1)
        print "FINAL RESULTS" + str('*' * 140)
        print self.A_p
        self.A_c.print_to(sys.stdout)

        for i in xrange(self.size - self.nbr_of_rows_to_delete):
            for j in xrange(self.size):
                if self.A_c[i,j] != self.A_p[i,j]:
                    print "(i,j) = (%d, %d)" % (i, j)
                    print self.A_c[i, j]
                    print self.A_p[i, j]
                assert self.A_c[i,j] == self.A_p[i,j]

    def test_pysparse(self):
        print "PySparse test"
        print self.A_p
        self.A_p.delete_rows(self.mask_p)
        time.sleep(1)
        print self.A_p
        print "End PySparse test"
        return

    def test_cysparse(self):
        print "CySparse test"
        #self.A_c.print_to(sys.stdout)
        self.A_c.delete_rows_with_mask(self.mask_c)
        time.sleep(1)
        #self.A_c.print_to(sys.stdout)
        print "End CySparse test"
        return

if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")