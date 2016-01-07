from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy


cdef class SumProxy(OpProxy):
    cdef:
        public bint __real_sum

