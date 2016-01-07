from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy


cdef class MulProxy(OpProxy):

    cdef bint operands_are_compatible(self, left, right)