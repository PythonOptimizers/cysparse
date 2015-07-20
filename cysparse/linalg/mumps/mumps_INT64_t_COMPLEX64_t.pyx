from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX64_t cimport LLSparseMatrix_INT64_t_COMPLEX64_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX64_t cimport CSCSparseMatrix_INT64_t_COMPLEX64_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF

import numpy as np
cimport numpy as cnp

cnp.import_array()

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

cdef extern from "cmumps_c.h":
    ctypedef struct CMUMPS_STRUC_C:
        MUMPS_INT      sym, par, job
        MUMPS_INT      comm_fortran    # Fortran communicator
        MUMPS_INT      icntl[40]
        MUMPS_INT      keep[500]
        CMUMPS_REAL    cntl[15]
        CMUMPS_REAL    dkeep[130];
        MUMPS_INT8     keep8[150];
        MUMPS_INT      n

        # used in matlab interface to decide if we
        # free + malloc when we have large variation
        MUMPS_INT      nz_alloc

        # Assembled entry
        MUMPS_INT      nz
        MUMPS_INT      *irn
        MUMPS_INT      *jcn
        CMUMPS_COMPLEX *a

        # Distributed entry
        MUMPS_INT      nz_loc
        MUMPS_INT      *irn_loc
        MUMPS_INT      *jcn_loc
        CMUMPS_COMPLEX *a_loc

        # Element entry
        MUMPS_INT      nelt
        MUMPS_INT      *eltptr
        MUMPS_INT      *eltvar
        CMUMPS_COMPLEX *a_elt

        # Ordering, if given by user
        MUMPS_INT      *perm_in

        # Orderings returned to user
        MUMPS_INT      *sym_perm    # symmetric permutation
        MUMPS_INT      *uns_perm    # column permutation

        # Scaling (input only in this version)
        CMUMPS_REAL    *colsca
        CMUMPS_REAL    *rowsca
        MUMPS_INT colsca_from_mumps;
        MUMPS_INT rowsca_from_mumps;


        # RHS, solution, ouptput data and statistics
        CMUMPS_COMPLEX *rhs
        CMUMPS_COMPLEX *redrhs
        CMUMPS_COMPLEX *rhs_sparse
        CMUMPS_COMPLEX *sol_loc
        MUMPS_INT      *irhs_sparse
        MUMPS_INT      *irhs_ptr
        MUMPS_INT      *isol_loc
        MUMPS_INT      nrhs, lrhs, lredrhs, nz_rhs, lsol_loc
        MUMPS_INT      schur_mloc, schur_nloc, schur_lld
        MUMPS_INT      mblock, nblock, nprow, npcol
        MUMPS_INT      info[40]
        MUMPS_INT      infog[40]
        CMUMPS_REAL    rinfo[40]
        CMUMPS_REAL    rinfog[40]

        # Null space
        MUMPS_INT      deficiency
        MUMPS_INT      *pivnul_list
        MUMPS_INT      *mapping

        # Schur
        MUMPS_INT      size_schur
        MUMPS_INT      *listvar_schur
        CMUMPS_COMPLEX *schur

        # Internal parameters
        MUMPS_INT      instance_number
        CMUMPS_COMPLEX *wk_user

        char *version_number
        # For out-of-core
        char *ooc_tmpdir
        char *ooc_prefix
        # To save the matrix in matrix market format
        char *write_problem
        MUMPS_INT      lwk_user





cdef class MumpsContext_INT64_t_COMPLEX64_t:
    """
    Mumps Context.

    This version **only** deals with ``LLSparseMatrix_INT64_t_COMPLEX64_t`` objects.

    We follow the common use of Mumps. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Mumps.
    """
    MUMPS_VERSION = 'TO BE DEFINED...'

    def __cinit__(self, LLSparseMatrix_INT64_t_COMPLEX64_t A):
        """
        Args:
            A: A :class:`LLSparseMatrix_INT64_t_COMPLEX64_t` object.

        Warning:
            The solver takes a "snapshot" of the matrix ``A``, i.e. the results given by the solver are only
            valid for the matrix given. If the matrix ``A`` changes aferwards, the results given by the solver won't
            reflect this change.

        """
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz
