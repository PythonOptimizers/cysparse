from cysparse.sparse.operator_proxies.op_matrix_proxy cimport OpMatrixProxy


cdef class SumProxy(OpMatrixProxy):
    cdef:
        public bint __real_sum

