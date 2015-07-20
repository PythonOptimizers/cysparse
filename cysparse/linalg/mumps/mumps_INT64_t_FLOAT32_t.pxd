from cysparse.types.cysparse_types cimport *

from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT32_t cimport LLSparseMatrix_INT64_t_FLOAT32_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT32_t cimport CSCSparseMatrix_INT64_t_FLOAT32_t

cimport numpy as cnp

cdef extern from "mumps_c_types.h":

    ctypedef int        MUMPS_INT
    ctypedef cnp.int8_t  MUMPS_INT8

    ctypedef float      SMUMPS_COMPLEX
    ctypedef float      SMUMPS_REAL

    ctypedef double     DMUMPS_COMPLEX
    ctypedef double     DMUMPS_REAL

    ctypedef struct mumps_complex:
        float r,i

    ctypedef mumps_complex  CMUMPS_COMPLEX
    ctypedef float          CMUMPS_REAL

    ctypedef struct mumps_double_complex:
        double r, i

    ctypedef mumps_double_complex  ZMUMPS_COMPLEX
    ctypedef double                ZMUMPS_REAL

cdef extern from "smumps_c.h":
    ctypedef struct SMUMPS_STRUC_C:
        MUMPS_INT      sym, par, job
        MUMPS_INT      comm_fortran    # Fortran communicator
        MUMPS_INT      icntl[40]
        MUMPS_INT      keep[500]
        SMUMPS_REAL    cntl[15]
        SMUMPS_REAL    dkeep[130];
        MUMPS_INT8     keep8[150];
        MUMPS_INT      n

        # used in matlab interface to decide if we
        # free + malloc when we have large variation
        MUMPS_INT      nz_alloc

        # Assembled entry
        MUMPS_INT      nz
        MUMPS_INT      *irn
        MUMPS_INT      *jcn
        SMUMPS_COMPLEX *a

        # Distributed entry
        MUMPS_INT      nz_loc
        MUMPS_INT      *irn_loc
        MUMPS_INT      *jcn_loc
        SMUMPS_COMPLEX *a_loc

        # Element entry
        MUMPS_INT      nelt
        MUMPS_INT      *eltptr
        MUMPS_INT      *eltvar
        SMUMPS_COMPLEX *a_elt

        # Ordering, if given by user
        MUMPS_INT      *perm_in

        # Orderings returned to user
        MUMPS_INT      *sym_perm    # symmetric permutation
        MUMPS_INT      *uns_perm    # column permutation

        # Scaling (input only in this version)
        SMUMPS_REAL    *colsca
        SMUMPS_REAL    *rowsca
        MUMPS_INT colsca_from_mumps;
        MUMPS_INT rowsca_from_mumps;


        # RHS, solution, ouptput data and statistics
        SMUMPS_COMPLEX *rhs
        SMUMPS_COMPLEX *redrhs
        SMUMPS_COMPLEX *rhs_sparse
        SMUMPS_COMPLEX *sol_loc
        MUMPS_INT      *irhs_sparse
        MUMPS_INT      *irhs_ptr
        MUMPS_INT      *isol_loc
        MUMPS_INT      nrhs, lrhs, lredrhs, nz_rhs, lsol_loc
        MUMPS_INT      schur_mloc, schur_nloc, schur_lld
        MUMPS_INT      mblock, nblock, nprow, npcol
        MUMPS_INT      info[40]
        MUMPS_INT      infog[40]
        SMUMPS_REAL    rinfo[40]
        SMUMPS_REAL    rinfog[40]

        # Null space
        MUMPS_INT      deficiency
        MUMPS_INT      *pivnul_list
        MUMPS_INT      *mapping

        # Schur
        MUMPS_INT      size_schur
        MUMPS_INT      *listvar_schur
        SMUMPS_COMPLEX *schur

        # Internal parameters
        MUMPS_INT      instance_number
        SMUMPS_COMPLEX *wk_user

        char *version_number
        # For out-of-core
        char *ooc_tmpdir
        char *ooc_prefix
        # To save the matrix in matrix market format
        char *write_problem
        MUMPS_INT      lwk_user

    cdef void smumps_c(SMUMPS_STRUC_C *)

cdef class MumpsContext_INT64_t_FLOAT32_t:
    cdef:
        LLSparseMatrix_INT64_t_FLOAT32_t A

        INT64_t nrow
        INT64_t ncol
        INT64_t nnz

        # Matrix A in CSC format
        CSCSparseMatrix_INT64_t_FLOAT32_t csc_mat

        SMUMPS_STRUC_C params

        # MUMPS
        cdef SMUMPS_STRUC_C ob