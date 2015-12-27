#!/usr/bin/env python

"""
This file tests internal memory for all matrices objects.

"""

import unittest
from cysparse.sparse.ll_mat import *


########################################################################################################################
# Tests
########################################################################################################################

NROW = 10
NCOL = 14
SIZE = 10


#######################################################################
# Case: store_symmetry == False, Store_zero==False
#######################################################################
class CySparseInternalMemoryNoSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_memory_real_in_bytes(self):
        """
        Approximative test for internal memory.

        As the exact memory needed might change over time (i.e. differ with different implementations), we only
        test for a lower bound. This test also the existence of the method.
        """
        nrow, ncol = self.C.shape
        nnz = self.C.nnz

        real_memory_index = self.C.memory_index_in_bytes()
        real_memory_type = self.C.memory_element_in_bytes()
        real_memory_matrix = self.C.memory_real_in_bytes()

{% if class == 'LLSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # root: nrow * sizeof(index)
        # col:  nnz * sizeof(index)
        # link: nnz * sizeof(index)
        # val:  nnz * sizeof(type)

        self.assertTrue(real_memory_matrix >= ((nrow + 2 * nnz) * real_memory_index + nnz * real_memory_type))
{% elif class == 'CSCSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # row:  nnz * sizeof(index)
        # val:  nnz * sizeof(type)
        # ind:  (ncol + 1) * sizeof(index)

        self.assertTrue(real_memory_matrix >= ((ncol + 1 + nnz) * real_memory_index + nnz * real_memory_type))
{% elif class == 'CSRSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # col:  nnz * sizeof(index)
        # val:  nnz * sizeof(type)
        # ind:  (nrow + 1) * sizeof(index)

        self.assertTrue(real_memory_matrix >= ((nrow + 1 + nnz) * real_memory_index + nnz * real_memory_type))
{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == True, Store_zero==False
#######################################################################
class CySparseInternalMemoryWithSymmetryNoZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_memory_real_in_bytes(self):
        """
        Approximative test for internal memory.

        As the exact memory needed might change over time (i.e. differ with different implementations), we only
        test for a lower bound. This test also the existence of the method.
        """
        nrow, ncol = self.C.shape
        nnz = self.C.nnz

        real_memory_index = self.C.memory_index_in_bytes()
        real_memory_type = self.C.memory_element_in_bytes()
        real_memory_matrix = self.C.memory_real_in_bytes()

{% if class == 'LLSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # root: nrow * sizeof(index)
        # col:  nnz * sizeof(index)
        # link: nnz * sizeof(index)
        # val:  nnz * sizeof(type)

        self.assertTrue(real_memory_matrix >= ((nrow + 2 * nnz) * real_memory_index + nnz * real_memory_type))
{% elif class == 'CSCSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # row:  nnz * sizeof(index)
        # val:  nnz * sizeof(type)
        # ind:  (ncol + 1) * sizeof(index)

        self.assertTrue(real_memory_matrix >= ((ncol + 1 + nnz) * real_memory_index + nnz * real_memory_type))
{% elif class == 'CSRSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # col:  nnz * sizeof(index)
        # val:  nnz * sizeof(type)
        # ind:  (nrow + 1) * sizeof(index)

        self.assertTrue(real_memory_matrix >= ((nrow + 1 + nnz) * real_memory_index + nnz * real_memory_type))
{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}


#######################################################################
# Case: store_symmetry == False, Store_zero==True
#######################################################################
class CySparseInternalMemoryNoSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.nrow = NROW
        self.ncol = NCOL

        self.A = LinearFillLLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=@type|type2enum@, itype=@index|type2enum@, store_zero=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_memory_real_in_bytes(self):
        """
        Approximative test for internal memory.

        As the exact memory needed might change over time (i.e. differ with different implementations), we only
        test for a lower bound. This test also the existence of the method.
        """
        nrow, ncol = self.C.shape
        nnz = self.C.nnz

        real_memory_index = self.C.memory_index_in_bytes()
        real_memory_type = self.C.memory_element_in_bytes()
        real_memory_matrix = self.C.memory_real_in_bytes()

{% if class == 'LLSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # root: nrow * sizeof(index)
        # col:  nnz * sizeof(index)
        # link: nnz * sizeof(index)
        # val:  nnz * sizeof(type)

        self.assertTrue(real_memory_matrix >= ((nrow + 2 * nnz) * real_memory_index + nnz * real_memory_type))
{% elif class == 'CSCSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # row:  nnz * sizeof(index)
        # val:  nnz * sizeof(type)
        # ind:  (ncol + 1) * sizeof(index)

        self.assertTrue(real_memory_matrix >= ((ncol + 1 + nnz) * real_memory_index + nnz * real_memory_type))
{% elif class == 'CSRSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # col:  nnz * sizeof(index)
        # val:  nnz * sizeof(type)
        # ind:  (nrow + 1) * sizeof(index)

        self.assertTrue(real_memory_matrix >= ((nrow + 1 + nnz) * real_memory_index + nnz * real_memory_type))
{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

#######################################################################
# Case: store_symmetry == True, Store_zero==True
#######################################################################
class CySparseInternalMemoryWithSymmetrySWithZero_@class@_@index@_@type@_TestCase(unittest.TestCase):
    def setUp(self):

        self.size = SIZE

        self.A = LinearFillLLSparseMatrix(size=self.size, dtype=@type|type2enum@, itype=@index|type2enum@, store_symmetry=True, store_zero=True)

{% if class == 'LLSparseMatrix' %}
        self.C = self.A

{% elif class == 'CSCSparseMatrix' %}
        self.C = self.A.to_csc()

{% elif class == 'CSRSparseMatrix' %}
        self.C = self.A.to_csr()

{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

    def test_memory_real_in_bytes(self):
        """
        Approximative test for internal memory.

        As the exact memory needed might change over time (i.e. differ with different implementations), we only
        test for a lower bound. This test also the existence of the method.
        """
        nrow, ncol = self.C.shape
        nnz = self.C.nnz

        real_memory_index = self.C.memory_index_in_bytes()
        real_memory_type = self.C.memory_element_in_bytes()
        real_memory_matrix = self.C.memory_real_in_bytes()

{% if class == 'LLSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # root: nrow * sizeof(index)
        # col:  nnz * sizeof(index)
        # link: nnz * sizeof(index)
        # val:  nnz * sizeof(type)

        self.assertTrue(real_memory_matrix >= ((nrow + 2 * nnz) * real_memory_index + nnz * real_memory_type))
{% elif class == 'CSCSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # row:  nnz * sizeof(index)
        # val:  nnz * sizeof(type)
        # ind:  (ncol + 1) * sizeof(index)

        self.assertTrue(real_memory_matrix >= ((ncol + 1 + nnz) * real_memory_index + nnz * real_memory_type))
{% elif class == 'CSRSparseMatrix' %}
        # As of December 2015, this bound is tight!
        # col:  nnz * sizeof(index)
        # val:  nnz * sizeof(type)
        # ind:  (nrow + 1) * sizeof(index)

        self.assertTrue(real_memory_matrix >= ((nrow + 1 + nnz) * real_memory_index + nnz * real_memory_type))
{% else %}
YOU SHOULD ADD YOUR NEW MATRIX CLASS HERE
{% endif %}

if __name__ == '__main__':
    unittest.main()

