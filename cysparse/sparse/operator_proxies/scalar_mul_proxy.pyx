from cysparse.sparse.operator_proxies.op_scalar_proxy cimport OpScalarProxy
from cysparse.sparse.operator_proxies.sum_proxy cimport SumProxy
from cysparse.sparse.operator_proxies.mul_proxy cimport MulProxy

from cysparse.common_types.cysparse_types import is_scalar

from cpython cimport Py_INCREF, Py_DECREF, PyObject
cdef extern from "Python.h":
    # *** Types ***
    int PyInt_Check(PyObject *o)

cimport numpy as cnp

cnp.import_array()

cdef class ScalarMulProxy(OpScalarProxy):
    def __cinit__(self, scalar, operand):
        pass


    ####################################################################################################################
    # Special methods
    ####################################################################################################################
    def __getitem__(self, tuple key):
        if len(key) != 2:
            raise IndexError('Index tuple must be of length 2 (not %d)' % len(key))

        if not PyInt_Check(<PyObject *>key[0]) or not PyInt_Check(<PyObject *>key[1]):
            raise IndexError("Only integers are accepted as indices for a Sum Proxy")

        return self.scalar * self.operand[key[0], key[1]]

    def __add__(self, other):
        return SumProxy(self, other)

    def __sub__(self, other):
        return SumProxy(self, other, real_sum=False)

    def __mul__(self, other):

        # if NumPy vector -> materialise
        if cnp.PyArray_Check(other) and  other.ndim == 1:
            return self.scalar * (self.operand * other)
        elif is_scalar(other):
            self.scalar *= other
            return self
        else:
            return MulProxy(self, other)
