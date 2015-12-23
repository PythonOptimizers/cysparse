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
class CySparseCommonNumpyVectorMultiplication_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.@type|type2enum|cysparse_type_to_numpy_type@)
        self.y = np.empty(self.nrow, dtype=np.@type|type2enum|cysparse_type_to_numpy_type@)
{% if type in complex_list %}
        self.x.fill(2 + 2j)
        self.y.fill(2 + 2j)
{% else %}
        self.x.fill(2)
        self.y.fill(2)
{% endif %}

{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
{% elif class == 'TransposedSparseMatrix' %}
        self.C = self.A.T
{% elif class == 'ConjugatedSparseMatrix' %}
        self.C = self.A.conj
{% elif class == 'ConjugateTransposedSparseMatrix' %}
        self.C = self.A.H
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}


    def test_numpy_vector_multiplication_element_by_element(self):
{% if class == 'TransposedSparseMatrix' or class == 'ConjugateTransposedSparseMatrix' %}
        result_with_A = self.A.T * self.y
        result_with_C = self.C * self.y

        for j in range(self.ncol):
            self.assertTrue(result_with_A[j] == result_with_C[j])

{% else %}
        result_with_A = self.A * self.x
        result_with_C = self.C * self.x

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])
{% endif %}

#########################################################################
# WITH STRIDES IN THE NUMPY VECTORS
#########################################################################

if __name__ == '__main__':
    unittest.main()