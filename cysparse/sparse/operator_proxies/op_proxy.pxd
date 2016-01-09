
# TODO: forbid direct creation of OpProxy?
# This is not absolutely necessary as an OpProxy needs two matrices to be created.

cdef class OpProxy:
    cdef:
        public int nrow
        public int ncol
        public int nnz

        public object left_operand
        public object right_operand

        public object dtype
        public object itype


    cdef bint operand_is_accepted(self, operand)
