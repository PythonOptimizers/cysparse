#!/usr/bin/env python

"""
This file tests the global ``matvec`` functions.

We tests:

- ``matvec()``;
- ``matvec_transp()``;
- ``matvec_htransp()``;
- ``matvec_conj()``;


"""

import unittest
from cysparse.sparse.ll_mat import *
from cysparse.common_types.cysparse_types import *

########################################################################################################################
# Tests
########################################################################################################################

############################################
# matvec
############################################
class CySparseGlobalMatVecFunction_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.@type|type2enum|cysparse_type_to_numpy_type@)
{% if type in complex_list %}
        self.x.fill(2 + 2j)
{% else %}
        self.x.fill(2)
{% endif %}

        # sparse vector
        self.v_ll_mat = LinearFillLLSparseMatrix(nrow=self.ncol, ncol=1, dtype=@type|type2enum@, itype=@index|type2enum@)

{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
        self.v = self.v_ll_mat.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
        self.v = self.v_ll_mat.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
        self.v = self.v_ll_mat
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec`` with a :program:`NumPy` vector.
        """
        result_with_A = self.A * self.x
        result_with_C = matvec(self.C, self.x)

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])

{% if class == 'LLSparseMatrix'%}
# TODO: enable this test with other matrix types when matdot is implemented for them
# ======================================================================================================================
    def test_sparse_matrix_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec`` with sparse vector.
        """
        result_with_A = self.A * self.v_ll_mat
        result_with_C = matvec(self.C, self.v)

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i,0] == result_with_C[i,0])


{% endif %}


############################################
# matvec_transp
############################################
class CySparseGlobalMatVecTranspFunction_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)

        # numpy vectors
        self.x = np.empty(self.nrow, dtype=np.@type|type2enum|cysparse_type_to_numpy_type@)
{% if type in complex_list %}
        self.x.fill(2 + 2j)
{% else %}
        self.x.fill(2)
{% endif %}

        # sparse vector
        self.v_ll_mat = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=1, dtype=@type|type2enum@, itype=@index|type2enum@)

{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
        self.v = self.v_ll_mat.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
        self.v = self.v_ll_mat.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
        self.v = self.v_ll_mat
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec_transp`` with a :program:`NumPy` vector.
        """
        result_with_A = self.A.T * self.x
        result_with_C = matvec_transp(self.C, self.x)

        for i in range(self.ncol):
            self.assertTrue(result_with_A[i] == result_with_C[i])

{% if class == 'LLSparseMatrix'%}
# TODO: enable this test with other matrix types when matdot is implemented for them
# ======================================================================================================================
    def test_sparse_matrix_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec_transp`` with a sparse vector.
        """
        result_with_A = self.A.T * self.v_ll_mat
        result_with_C = matvec_transp(self.C, self.v)

        for i in range(self.ncol):
            self.assertTrue(result_with_A[i,0] == result_with_C[i,0])


{% endif %}

############################################
# matvec_htransp
############################################
class CySparseGlobalMatVecHTranspFunction_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)

        # numpy vectors
        self.x = np.empty(self.nrow, dtype=np.@type|type2enum|cysparse_type_to_numpy_type@)
{% if type in complex_list %}
        self.x.fill(2 + 2j)
{% else %}
        self.x.fill(2)
{% endif %}

{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec_htransp`` with a :program:`NumPy` vector.
        """
        result_with_A = self.A.H * self.x
        result_with_C = matvec_htransp(self.C, self.x)

        for i in range(self.ncol):
            self.assertTrue(result_with_A[i] == result_with_C[i])


############################################
# matvec_conj
############################################
class CySparseGlobalMatVecConjFunction_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):
        self.nrow = 10
        self.ncol = 14

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)

        # numpy vectors
        self.x = np.empty(self.ncol, dtype=np.@type|type2enum|cysparse_type_to_numpy_type@)
{% if type in complex_list %}
        self.x.fill(2 + 2j)
{% else %}
        self.x.fill(2)
{% endif %}

{% if class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()
{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()
{% elif class == 'LLSparseMatrix'%}
        self.C = self.A
{% else %}
YOU HAVE TO ADD YOUR NEW MATRIX TYPE HERE
{% endif %}

# ======================================================================================================================
    def test_numpy_vector_multiplication_element_by_element(self):
        """
        Test global ``matvec_conj`` with a :program:`NumPy` vector.
        """
        result_with_A = self.A.conj * self.x
        result_with_C = matvec_conj(self.C, self.x)

        for i in range(self.nrow):
            self.assertTrue(result_with_A[i] == result_with_C[i])


if __name__ == '__main__':
    unittest.main()