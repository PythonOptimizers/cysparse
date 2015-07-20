from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT32_t cimport LLSparseMatrix_INT64_t_FLOAT32_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT32_t cimport CSCSparseMatrix_INT64_t_FLOAT32_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF

import numpy as np
cimport numpy as cnp

from libc.stdint cimport int64_t
from libc.string cimport strncpy

cnp.import_array()


cdef extern from "mumps_c_types.h":

    ctypedef int        MUMPS_INT
    ctypedef int64_t  MUMPS_INT8 # warning: mumps uses "stdint.h" which might define int64_t as long long...

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


cdef class mumps_int_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters
    int version.

    """
    def __cinit__(self):
        pass

    cdef get_array(self, MUMPS_INT * array, int ub = 40):
        """
        Args:
            ub: upper bound.
        """
        self.ub = ub
        self.array = array

    def __getitem__(self, key):
        if key < 1:
            raise IndexError('Mumps index must be >= 1 (Fortran style)')
        if key > self.ub:
            raise IndexError('Mumps index must be <= %d' % self.ub)

        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value

cdef class smumps_real_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters
    Real version.

    """
    def __cinit__(self):
        pass

    cdef get_array(self, SMUMPS_REAL * array, int ub = 40):
        """
        Args:
            ub: upper bound.
        """
        self.ub = ub
        self.array = array

    def __getitem__(self, key):
        if key < 1:
            raise IndexError('Mumps index must be >= 1 (Fortran style)')
        if key > self.ub:
            raise IndexError('Mumps index must be <= %d' % self.ub)

        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value










cdef class MumpsContext_INT64_t_FLOAT32_t:
    """
    Mumps Context.

    This version **only** deals with ``LLSparseMatrix_INT64_t_FLOAT32_t`` objects.

    We follow the common use of Mumps. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Mumps.
    """

    def __cinit__(self, LLSparseMatrix_INT64_t_FLOAT32_t A, comm_fortran=-987654):
        """
        Args:
            A: A :class:`LLSparseMatrix_INT64_t_FLOAT32_t` object.

        Warning:
            The solver takes a "snapshot" of the matrix ``A``, i.e. the results given by the solver are only
            valid for the matrix given. If the matrix ``A`` changes aferwards, the results given by the solver won't
            reflect this change.

        """
        assert A.ncol == A.nrow

        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz

        # MUMPS
        self.params.job = -1
        self.params.sym = self.A.is_symmetric
        self.params.par = 1

        self.params.comm_fortran = comm_fortran

        smumps_c(&self.params)

        # mumps int arrays
        # integer control parameters
        self.icntl = mumps_int_array()
        self.icntl.get_array(self.params.icntl)

        self.set_silent()

        self.info = mumps_int_array()
        self.info.get_array(self.params.info)

        self.infog = mumps_int_array()
        self.infog.get_array(self.params.infog)

        # real control parameters
        self.cntl = smumps_real_array()
        self.cntl.get_array(self.params.cntl)

        self.rinfo = smumps_real_array()
        self.rinfo.get_array(self.params.rinfo)

        self.rinfog = smumps_real_array()
        self.rinfog.get_array(self.params.rinfog)

        # create i, j, val
        self.A.take_triplet_pointers(self.a_row, self.a_col, self.a_val)

        self.params.n = <MUMPS_INT> self.nrow

        self.set_centralized_assembled_matrix()


    cdef set_centralized_assembled_matrix(self):
        """
        Set the centralized assembled matrix
        The rank 0 process supplies the entire matrix.
        """

        self.params.nz = <MUMPS_INT> self.nnz

        self.params.irn = <MUMPS_INT *> self.a_row
        self.params.jcn = <MUMPS_INT *> self.a_col
        self.params.a = <SMUMPS_COMPLEX *> self.a_val


    def __dealloc__(self):
        # autodestruct mumps internal
        self.params.job = -2
        self.mumps_call()

        PyMem_Free(self.a_row)
        PyMem_Free(self.a_col)
        PyMem_Free(self.a_val)

    ####################################################################################################################
    # Properties
    ####################################################################################################################
    ######################################### COMMON Properties ########################################################
    property sym:
        def __get__(self): return self.params.sym
        def __set__(self, value): self.params.sym = value
    property par:
        def __get__(self): return self.params.par
        def __set__(self, value): self.params.par = value
    property job:
        def __get__(self): return self.params.job
        def __set__(self, value): self.params.job = value

    property comm_fortran:
        def __get__(self): return self.params.comm_fortran
        def __set__(self, value): self.params.comm_fortran = value

    #XXXXXXXXXXXX
    property icntl:
        def __get__(self):
            return self.icntl

    #property icntl:
    #    def __get__(self):
    #        cdef MUMPS_INT[::1] view = <MUMPS_INT[::1]> self.params.icntl
    #        return view

    property n:
        def __get__(self): return self.params.n
        def __set__(self, value): self.params.n = value
    property nz_alloc:
        def __get__(self): return self.params.nz_alloc
        def __set__(self, value): self.params.nz_alloc = value

    property nz:
        def __get__(self): return self.params.nz
        def __set__(self, value): self.params.nz = value
    property irn:
        def __get__(self): return <long> self.params.irn
        def __set__(self, long value): self.params.irn = <MUMPS_INT*> value
    property jcn:
        def __get__(self): return <long> self.params.jcn
        def __set__(self, long value): self.params.jcn = <MUMPS_INT*> value

    property nz_loc:
        def __get__(self): return self.params.nz_loc
        def __set__(self, value): self.params.nz_loc = value
    property irn_loc:
        def __get__(self): return <long> self.params.irn_loc
        def __set__(self, long value): self.params.irn_loc = <MUMPS_INT*> value
    property jcn_loc:
        def __get__(self): return <long> self.params.jcn_loc
        def __set__(self, long value): self.params.jcn_loc = <MUMPS_INT*> value

    property nelt:
        def __get__(self): return self.params.nelt
        def __set__(self, value): self.params.nelt = value
    property eltptr:
        def __get__(self): return <long> self.params.eltptr
        def __set__(self, long value): self.params.eltptr = <MUMPS_INT*> value
    property eltvar:
        def __get__(self): return <long> self.params.eltvar
        def __set__(self, long value): self.params.eltvar = <MUMPS_INT*> value

    property perm_in:
        def __get__(self): return <long> self.params.perm_in
        def __set__(self, long value): self.params.perm_in = <MUMPS_INT*> value

    property sym_perm:
        def __get__(self): return <long> self.params.sym_perm
        def __set__(self, long value): self.params.sym_perm = <MUMPS_INT*> value
    property uns_perm:
        def __get__(self): return <long> self.params.uns_perm
        def __set__(self, long value): self.params.uns_perm = <MUMPS_INT*> value

    property irhs_sparse:
        def __get__(self): return <long> self.params.irhs_sparse
        def __set__(self, long value): self.params.irhs_sparse = <MUMPS_INT*> value
    property irhs_ptr:
        def __get__(self): return <long> self.params.irhs_ptr
        def __set__(self, long value): self.params.irhs_ptr = <MUMPS_INT*> value
    property isol_loc:
        def __get__(self): return <long> self.params.isol_loc
        def __set__(self, long value): self.params.isol_loc = <MUMPS_INT*> value

    property nrhs:
        def __get__(self): return self.params.nrhs
        def __set__(self, value): self.params.nrhs = value
    property lrhs:
        def __get__(self): return self.params.lrhs
        def __set__(self, value): self.params.lrhs = value
    property lredrhs:
        def __get__(self): return self.params.lredrhs
        def __set__(self, value): self.params.lredrhs = value
    property nz_rhs:
        def __get__(self): return self.params.nz_rhs
        def __set__(self, value): self.params.nz_rhs = value
    property lsol_loc:
        def __get__(self): return self.params.lsol_loc
        def __set__(self, value): self.params.lsol_loc = value

    property schur_mloc:
        def __get__(self): return self.params.schur_mloc
        def __set__(self, value): self.params.schur_mloc = value
    property schur_nloc:
        def __get__(self): return self.params.schur_nloc
        def __set__(self, value): self.params.schur_nloc = value
    property schur_lld:
        def __get__(self): return self.params.schur_lld
        def __set__(self, value): self.params.schur_lld = value


    property mblock:
        def __get__(self): return self.params.mblock
        def __set__(self, value): self.params.mblock = value
    property nblock:
        def __get__(self): return self.params.nblock
        def __set__(self, value): self.params.nblock = value
    property nprow:
        def __get__(self): return self.params.nprow
        def __set__(self, value): self.params.nprow = value
    property npcol:
        def __get__(self): return self.params.npcol
        def __set__(self, value): self.params.npcol = value

    property info:
        def __get__(self):
            cdef MUMPS_INT[::1] view = <MUMPS_INT[::1]> self.params.info
            return view
    property infog:
        def __get__(self):
            cdef MUMPS_INT[::1] view = <MUMPS_INT[::1]> self.params.infog
            return view

    property deficiency:
        def __get__(self): return self.params.deficiency
        def __set__(self, value): self.params.deficiency = value
    property pivnul_list:
        def __get__(self): return <long> self.params.pivnul_list
        def __set__(self, long value): self.params.pivnul_list = <MUMPS_INT*> value
    property mapping:
        def __get__(self): return <long> self.params.mapping
        def __set__(self, long value): self.params.mapping = <MUMPS_INT*> value

    property size_schur:
        def __get__(self): return self.params.size_schur
        def __set__(self, value): self.params.size_schur = value
    property listvar_schur:
        def __get__(self): return <long> self.params.listvar_schur
        def __set__(self, long value): self.params.listvar_schur = <MUMPS_INT*> value

    property instance_number:
        def __get__(self): return self.params.instance_number
        def __set__(self, value): self.params.instance_number = value

    property version_number:
        def __get__(self):
            return (<bytes> self.params.version_number).decode('ascii')

    property ooc_tmpdir:
        def __get__(self):
            return (<bytes> self.params.ooc_tmpdir).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.params.ooc_tmpdir, value, sizeof(self.params.ooc_tmpdir))
    property ooc_prefix:
        def __get__(self):
            return (<bytes> self.params.ooc_prefix).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.params.ooc_prefix, value, sizeof(self.params.ooc_prefix))

    property write_problem:
        def __get__(self):
            return (<bytes> self.params.write_problem).decode('ascii')
        def __set__(self, char *value):
            strncpy(self.params.write_problem, value, sizeof(self.params.write_problem))

    property lwk_user:
        def __get__(self): return self.params.lwk_user
        def __set__(self, value): self.params.lwk_user = value

    ######################################### TYPED Properties #########################################################
    property cntl:
        def __get__(self):
            cdef SMUMPS_REAL[::1] view = <SMUMPS_REAL[::1]> self.params.cntl
            return view

    property a:
        def __get__(self): return <long> self.params.a
        def __set__(self, long value): self.params.a = <SMUMPS_COMPLEX*> value

    property a_loc:
        def __get__(self): return <long> self.params.a_loc
        def __set__(self, long value): self.params.a_loc = <SMUMPS_COMPLEX*> value

    property a_elt:
        def __get__(self): return <long> self.params.a_elt
        def __set__(self, long value): self.params.a_elt = <SMUMPS_COMPLEX*> value

    property colsca:
        def __get__(self): return <long> self.params.colsca
        def __set__(self, long value): self.params.colsca = <SMUMPS_REAL*> value
    property rowsca:
        def __get__(self): return <long> self.params.rowsca
        def __set__(self, long value): self.params.rowsca = <SMUMPS_REAL*> value

    property rhs:
        def __get__(self): return <long> self.params.rhs
        def __set__(self, long value): self.params.rhs = <SMUMPS_COMPLEX*> value
    property redrhs:
        def __get__(self): return <long> self.params.redrhs
        def __set__(self, long value): self.params.redrhs = <SMUMPS_COMPLEX*> value
    property rhs_sparse:
        def __get__(self): return <long> self.params.rhs_sparse
        def __set__(self, long value): self.params.rhs_sparse = <SMUMPS_COMPLEX*> value
    property sol_loc:
        def __get__(self): return <long> self.params.sol_loc
        def __set__(self, long value): self.params.sol_loc = <SMUMPS_COMPLEX*> value

    property rinfo:
        def __get__(self):
            cdef SMUMPS_REAL[::1] view = <SMUMPS_REAL[::1]> self.params.rinfo
            return view
    property rinfog:
        def __get__(self):
            cdef SMUMPS_REAL[::1] view = <SMUMPS_REAL[::1]> self.params.rinfog
            return view

    property schur:
        def __get__(self): return <long> self.params.schur
        def __set__(self, long value): self.params.schur = <SMUMPS_COMPLEX*> value

    property wk_user:
        def __get__(self): return <long> self.params.wk_user
        def __set__(self, long value): self.params.wk_user = <SMUMPS_COMPLEX*> value

    ####################################################################################################################
    # MUMPS CALL
    ####################################################################################################################
    cdef mumps_call(self):
        """
        Call to Xmumps_c(XMUMPS_STRUC_C).
        """
        smumps_c(&self.params)


    def set_silent(self):
        """
        Silence **all* Mumps output.

        See Mumps documentation.
        """
        self.icntl[1] = 0
        self.icntl[2] = 0
        self.icntl[3] = 0
        self.icntl[4] = 0

    cdef set_dense_rhs(self, FLOAT32_t * rhs, INT64_t rhs_length, INT64_t nrhs):
        """
        Args:
            rhs: Matrix with ``rhs`` member(s).
            rhs_length: Length of each column of the ``rhs``.
            nrhs: Number of columns in the matrix ``rhs``.
        """

        self.params.nrhs = <MUMPS_INT> nrhs
        self.params.lrhs = <MUMPS_INT> rhs_length
        self.params.rhs = <FLOAT32_t *>rhs

    #cdef set_sparse_rhs(self,
    #                   np.ndarray[MUMPS_INT, ndim=1] col_ptr,
    #                   np.ndarray[MUMPS_INT, ndim=1] row_ind,
    #                   np.ndarray[np.float64_t, ndim=1] data):

    #    if row_ind.shape[0] != data.shape[0]:
    #        raise ValueError("Number of entries in row index and value "
    #                         "array differ!")

    #    self.params.nz_rhs = data.shape[0]
    #    self.params.nrhs = col_ptr.shape[0] - 1
    #    self.params.rhs_sparse = <DMUMPS_COMPLEX *>data.data
    #    self.params.irhs_sparse = <MUMPS_INT *>row_ind.data
    #    self.params.irhs_ptr = <MUMPS_INT *>col_ptr.data

