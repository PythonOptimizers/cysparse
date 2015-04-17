from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

cdef extern from "suitesparse/umfpack.h":
    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO

    # OPAQUE UMFPACK OBJECTS
    cdef struct Symbolic:
        pass
    ctypedef Symbolic * symbolic_t

    cdef struct Numeric:
        pass
    ctypedef Numeric * numeric_t


cdef class UmfpackSolver:
    cdef:
        LLSparseMatrix A

        int nrow
        int ncol

        # Matrix A in CSC format
        cdef double * val
        cdef int * col
        cdef int * ind

        # UMFPACK opaque objects
        symbolic_t symbolic
        bint symbolic_computed

        numeric_t numeric
        bint numeric_computed

        # Control and Info arrays
        public double info[UMFPACK_INFO]
        public double control[UMFPACK_CONTROL]

    cdef create_symbolic(self, recompute=?)