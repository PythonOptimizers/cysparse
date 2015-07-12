from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT64_t cimport LLSparseMatrix_INT32_t_FLOAT64_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT32_t_FLOAT64_t cimport CSCSparseMatrix_INT32_t_FLOAT64_t


cdef extern from "umfpack.h":
    cdef enum:
        UMFPACK_CONTROL, UMFPACK_INFO


cdef class UmfpackSolver_INT32_t_FLOAT64_t:
    cdef:
        LLSparseMatrix_INT32_t_FLOAT64_t A

        INT32_t nrow
        INT32_t ncol

        # Matrix A in CSC format
        CSCSparseMatrix_INT32_t_FLOAT64_t csc_mat

        # UMFPACK opaque objects
        void * symbolic
        bint symbolic_computed

        void * numeric
        bint numeric_computed

        # Control and Info arrays
        public double info[UMFPACK_INFO]
        public double control[UMFPACK_CONTROL]

    cdef int _create_symbolic(self)
    cdef int _create_numeric(self)
