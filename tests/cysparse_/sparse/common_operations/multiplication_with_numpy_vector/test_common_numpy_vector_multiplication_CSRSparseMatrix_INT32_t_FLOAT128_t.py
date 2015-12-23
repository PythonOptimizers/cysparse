#!/usr/bin/env python

"""
This file tests the basic common multiplication operations with a :program:`NumPy` vector for **all** matrix like objects.

Proxies are only tested for a :class:`LLSparseMatrix` object.

We tests:

- Matrix-like objects:
    * with/without Symmetry;
    * with/without StoreZero scheme

- Numpy vectors:
    * with/without strides

See file ``sparse_matrix_like_vector_multiplication``.
"""
from sparse_matrix_like_vector_multiplication import common_matrix_like_vector_multiplication

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

#########################################################################
# WITHOUT STRIDES IN THE NUMPY VECTORS
#########################################################################
##################################
# Case Non Symmetric, Non Zero
##################################
class CySparseCommonNumpyVectorMultiplication_CSRSparseMatrix_INT32_t_FLOAT128_t_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=FLOAT128_T, itype=INT32_T)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.float128)
        self.y = np.empty(self.nrow, dtype=np.float128)

        self.x.fill(2)
        self.y.fill(2)



        self.C = self.A.to_csr()



    def test_numpy_vector_multiplication_element_by_element(self):

        result_with_A = self.A * self.x
        result_with_C = self.C * self.x

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])


#########################################################################
# WITH STRIDES IN THE NUMPY VECTORS
#########################################################################

if __name__ == '__main__':
    unittest.main()