from cysparse.sparse.s_mat cimport PySparseMatrix_Check

cdef class OpProxy:
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

        self.dtype = self.left_operand.dtype
        self.itype = self.left_operand.itype

    ####################################################################################################################
    # Helper methods
    ####################################################################################################################
    cdef bint operand_is_accepted(self, operand):
        cdef bint is_accepted = False

        if isinstance(operand, OpProxy):
            is_accepted = True
        elif PySparseMatrix_Check(operand):
            is_accepted = True

        # TODO: complete with SparseProxies, Views

        return is_accepted

    ####################################################################################################################
    # Special methods
    ####################################################################################################################
    def __getitem__(self, tuple key):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()

    def __sub__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    ####################################################################################################################
    # Materialisation methods
    ####################################################################################################################
    def to_ndarray(self):
        raise NotImplementedError()

    def to_ll(self):
        raise NotImplementedError()
    
    def to_csr(self):
        raise NotImplementedError()

    def to_csc(self):
        raise NotImplementedError()