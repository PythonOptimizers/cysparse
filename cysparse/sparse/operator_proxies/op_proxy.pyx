from cysparse.sparse.s_mat cimport PySparseMatrix_Check, \
                                   PyTransposedSparseMatrix_Check, \
                                   PyConjugatedSparseMatrix_Check, \
                                   PyConjugateTransposedSparseMatrix_Check, \
                                   PyLLSparseMatrixView_Check


cdef class OpProxy:
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

    ####################################################################################################################
    # Helper methods
    ####################################################################################################################
    cdef bint operand_is_accepted(self, operand):
        cdef bint is_accepted = False

        if isinstance(operand, OpProxy):
            is_accepted = True
        elif PySparseMatrix_Check(operand):
            is_accepted = True
        elif PyTransposedSparseMatrix_Check(operand) or PyConjugatedSparseMatrix_Check(operand) or PyConjugateTransposedSparseMatrix_Check(operand):
            is_accepted = True
        elif PyLLSparseMatrixView_Check(operand):
            is_accepted = True

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
        raise NotImplementedError("Transform first to a LLSparseMatrix and then call 'to_ndarray()'")

    def to_ll(self):
        from cysparse.sparse.ll_mat import LLSparseMatrix
        A = LLSparseMatrix(nrow=self.nrow, ncol=self.ncol, dtype=self.dtype, itype=self.itype, store_zero=False, size_hint=self.left_operand.nnz)
        for i in xrange(self.nrow):
            for j in xrange(self.ncol):
                A[i, j] = self[i, j]

        return A

    def to_csr(self):
        raise NotImplementedError("Transform first to a LLSparseMatrix and then call 'to_csr()'")

    def to_csc(self):
        raise NotImplementedError("Transform first to a LLSparseMatrix and then call 'to_csr()'")

