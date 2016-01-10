from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy
from cysparse.sparse.operator_proxies.mul_proxy cimport MulProxy

from cpython cimport Py_INCREF, Py_DECREF, PyObject
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)

cimport numpy as cnp

cnp.import_array()

cdef class SumProxy(OpProxy):
    def __cinit__(self, left_operand, right_operand, bint real_sum=True):

        assert left_operand.nrow == right_operand.nrow and left_operand.ncol == right_operand.ncol,\
            "Dimensions must be compatible [%d, %d] + [%d, %d]" % \
            (left_operand.nrow, left_operand.ncol, right_operand.nrow, right_operand.ncol)

        # sum or difference?
        self.__real_sum = real_sum

    ####################################################################################################################
    # Helper methods
    ####################################################################################################################


    ####################################################################################################################
    # Special methods
    ####################################################################################################################
    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integers are accepted as indices for a Sum Proxy")

        if self.__real_sum:
            return self.left_operand[key[0], key[1]] + self.right_operand[key[0], key[1]]
        else:
            return self.left_operand[key[0], key[1]] - self.right_operand[key[0], key[1]]

    def __add__(self, other):
        return SumProxy(self, other)

    def __sub__(self, other):
        return SumProxy(self, other, real_sum=False)

    def __mul__(self, other):

        # if NumPy vector -> materialise
        if cnp.PyArray_Check(other) and  other.ndim == 1:
            if self.__real_sum:
                return self.left_operand * other + self.right_operand * other
            else:
                return self.left_operand * other - self.right_operand * other
        else:
            return MulProxy(self, other)

