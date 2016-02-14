from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy


cdef class OpScalarProxy(OpProxy):
    cdef:
        public object scalar
        public object operand

