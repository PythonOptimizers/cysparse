from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy
from cysparse.sparse.operator_proxies.mul_proxy cimport MulProxy

from python_ref cimport Py_INCREF, Py_DECREF, PyObject
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)

cimport numpy as cnp

cnp.import_array()

cdef class SumProxy(OpProxy):
    def __cinit__(self, left_operand, right_operand, bint real_sum=True):

        assert self.operands_are_compatible(left_operand, right_operand), "Operands don't match"

        # sum or difference?
        self.real_sum = real_sum

    ####################################################################################################################
    # Helper methods
    ####################################################################################################################
    cdef bint operands_are_compatible(self, left, right):
        return left.nrow == right.nrow and left.ncol == right.ncol


    ####################################################################################################################
    # Special methods
    ####################################################################################################################
    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integers are accepted as indices for a Sum Proxy")

        return self.left_operand[key[0], key[1]] + self.right_operand[key[0], key[1]]

    def __add__(self, other):
        return SumProxy(self, other)

    def __sub__(self, other):
        return SumProxy(self, other, real_sum=False)

    def __mul__(self, other):
        # if NumPy vector -> materialise
        if cnp.PyArray_Check(other) and  other.ndim == 1:
            return self.left_operand * other + self.right_operand * other
        else:
            return MulProxy(self, other)
