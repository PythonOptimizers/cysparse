from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy


cdef class OpMatrixProxy(OpProxy):
    cdef:
        public object left_operand
        public object right_operand


