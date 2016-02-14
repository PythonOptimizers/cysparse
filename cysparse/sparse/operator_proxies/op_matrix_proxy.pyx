from cysparse.sparse.operator_proxies.op_proxy cimport OpProxy


cdef class OpMatrixProxy(OpProxy):
    """
    Generic Operator Proxy class.

    Warning:
        **Only** works with operation like matrix-addition, -substration and -multiplication. They all share some common
        ground that we use here.
    """
    def __cinit__(self, left_operand, right_operand, *args, **kwargs):

        assert self.operand_is_accepted(left_operand), "Operand %s is not accepted for this operator" % type(left_operand)
        assert self.operand_is_accepted(right_operand), "Operand %s is not accepted for this operator" % type(right_operand)

        # test of itype and dtype
        assert left_operand.dtype == right_operand.dtype, "Both operands must share the same dtype"
        assert left_operand.itype == right_operand.itype, "Both operands must share the same itype"

        self.left_operand = left_operand
        self.right_operand = right_operand

        self.nrow = self.left_operand.nrow
        self.ncol = self.right_operand.ncol
        self.nnz = self.left_operand.nnz

        self.dtype = self.left_operand.dtype
        self.itype = self.left_operand.itype




