from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy


cdef class SumProxy(OpProxy):
    cdef:
        bint real_sum

        bint nonexisting

    cdef bint operands_are_compatible(self, left, right)