from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy


cdef class OpScalarProxy(OpProxy):
    def __cinit__(self, scalar, operand, *args, **kwargs):

        assert self.operand_is_accepted(operand), "Operand %s is not accepted for this operator" % type(operand)
        assert self.scalar_is_accepted(scalar), "Scalar %s is not accepted for this operator" % type(scalar)

        self.scalar = scalar
        self.operand = operand

        self.nrow = self.operand.nrow
        self.ncol = self.operand.ncol
        self.nnz = self.operand.nnz

        self.dtype = self.operand.dtype
        self.itype = self.operand.itype



