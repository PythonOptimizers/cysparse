from cysparse.sparse.operator_proxies.op_matrix_proxy cimport OpMatrixProxy
from cysparse.sparse.operator_proxies.sum_proxy cimport SumProxy

from cpython cimport Py_INCREF, Py_DECREF, PyObject
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)

cimport numpy as cnp

cnp.import_array()

cdef class MulProxy(OpMatrixProxy):
    def __cinit__(self, left_operand, right_operand):
        # if isinstance(left_operand, OpMatrixProxy) and isinstance(right_operand, OpMatrixProxy):
        assert left_operand.ncol == right_operand.nrow, \
            "Dimensions must be compatible [%d, %d] * [%d, %d]" % \
            (left_operand.nrow, left_operand.ncol, right_operand.nrow, right_operand.ncol)

    ####################################################################################################################
    # Helper methods
    ####################################################################################################################


    ####################################################################################################################
    # Special methods
    ####################################################################################################################
    def __getitem__(self, tuple key):
        """

        Note:
            This method is **extremely** costly.

        """
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integers are accepted as indices for a Sum Proxy")

        # if isinstance(left_operand, OpMatrixProxy) and isinstance(right_operand, OpMatrixProxy):
        sum = 0

        p = self.left_operand.ncol


        for k in xrange(p):
            sum += self.left_operand[key[0], k] * self.right_operand[k, key[1]]

        return sum

    def __add__(self, other):
        return SumProxy(self, other)

    def __sub__(self, other):
        return SumProxy(self, other, real_sum=False)

    def __mul__(self, other):
        # if NumPy vector -> materialise
        if cnp.PyArray_Check(other) and  other.ndim == 1:
            return self.left_operand * (self.right_operand * other)
        else:
            return MulProxy(self, other)

