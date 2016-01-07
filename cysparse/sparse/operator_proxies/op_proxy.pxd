
# TODO: forbid direct creation of OpProxy?

cdef class OpProxy:
    cdef:
        public int nrow
        public int ncol

        public object left_operand
        public object right_operand

        public object dtype
        public object itype


    cdef bint operand_is_accepted(self, operand)
