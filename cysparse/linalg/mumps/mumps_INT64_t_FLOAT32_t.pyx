from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT32_t cimport LLSparseMatrix_INT64_t_FLOAT32_t
from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_FLOAT32_t cimport CSCSparseMatrix_INT64_t_FLOAT32_t

from cysparse.types.cysparse_numpy_types import are_mixed_types_compatible, cysparse_to_numpy_type

from cysparse.linalg.mumps.mumps_statistics import AnalysisStatistics, FactorizationStatistics, SolveStatistics

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF

from libc.stdint cimport int64_t
from libc.string cimport strncpy

import numpy as np
cimport numpy as cnp

cnp.import_array()

import time

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

########################################################################################################################
# MUMPS
########################################################################################################################
orderings = { 'amd' : 0, 'amf' : 2, 'scotch' : 3, 'pord' : 4, 'metis' : 5,
              'qamd' : 6, 'auto' : 7 }

ordering_name = [ 'amd', 'user-defined', 'amf',
                  'scotch', 'pord', 'metis', 'qamd']

################################################################
# MUMPS ERRORS
################################################################
# TODO: decouple
error_messages = {
    -5 : "Not enough memory during analysis phase",
    -6 : "Matrix is singular in structure",
    -7 : "Not enough memory during analysis phase",
    -10 : "Matrix is numerically singular",
    -11 : "The authors of MUMPS would like to hear about this",
    -12 : "The authors of MUMPS would like to hear about this",
    -13 : "Not enough memory"
}


class MUMPSError(RuntimeError):
    def __init__(self, infog):
        self.error = infog[1]
        if self.error in error_messages:
            msg = "{}. (MUMPS error {})".format(
                error_messages[self.error], self.error)
        else:
            msg = "MUMPS failed with error {}.".format(self.error)

        RuntimeError.__init__(self, msg)

################################################################
# MUMPS HELPERS
################################################################
cdef class mumps_int_array:
    """
    Internal classes to use x[i] = value and x[i] setters and getters

    Integer version.

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





cdef c_to_fortran_index_array(INT64_t * a, INT64_t a_size):
    cdef:
        INT64_t i

    for i from 0 <= i < a_size:
        a[i] += 1

################################################################
# MUMPS CONTEXT
################################################################
cdef class MumpsContext_INT64_t_FLOAT32_t:
    """
    Mumps Context.

    This version **only** deals with ``LLSparseMatrix_INT64_t_FLOAT32_t`` objects.

    We follow the common use of Mumps. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Mumps.
    """

    def __cinit__(self, LLSparseMatrix_INT64_t_FLOAT32_t A, comm_fortran=-987654, verbose=False):
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
        self.analysed = False
        self.factored = False
        self.out_of_core = False

        self.params.job = -1
        self.params.sym = self.A.is_symmetric
        self.params.par = 1

        self.params.comm_fortran = comm_fortran

        smumps_c(&self.params)

        # mumps int arrays
        # integer control parameters
        self.icntl = mumps_int_array()
        self.icntl.get_array(self.params.icntl)

        if not verbose:
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
        #cdef:
        #    INT64_t * a_row
        #    INT64_t * a_col
        #    FLOAT32_t *  a_val

        self.a_row = <INT64_t *> PyMem_Malloc(self.nnz * sizeof(INT64_t))
        self.a_col = <INT64_t *> PyMem_Malloc(self.nnz * sizeof(INT64_t))
        self.a_val = <FLOAT32_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT32_t))

        self.A.take_triplet_pointers(self.a_row, self.a_col, self.a_val)

        # transform c index arrays to fortran arrays
        c_to_fortran_index_array(self.a_row, self.nnz)
        c_to_fortran_index_array(self.a_col, self.nnz)

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

    property icntl:
        def __get__(self):
            return self.icntl

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
            return self.info

    property infog:
        def __get__(self):
            return self.infog

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
            return self.cntl


    #property cntl:
    #    def __get__(self):
    #        cdef SMUMPS_REAL[::1] view = <SMUMPS_REAL[::1]> self.params.cntl
    #        return view

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
            return self.rinfo

    property rinfog:
        def __get__(self):
            return self.rinfog

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



    ####################################################################################################################
    # Analyse
    ####################################################################################################################
    def analyse(self, ordering='auto'):
        """
        Perform analysis step of MUMPS.

        [TO BE REWRITTEN: Sylvain]

        In the analyis step, MUMPS figures out a reordering for the matrix and
        estimates number of operations and memory needed for the factorization
        time. This step usually needs not be called separately (it is done
        automatically by `factor`), but it can be useful to test which ordering
        would give best performance in the actual factorization, as MUMPS
        estimates are available in `analysis_stats`.

        Args:
            ordering : { 'auto', 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd' }
                ordering to use in the factorization. The availability of a
                particular ordering depends on the MUMPS installation.  Default is
                'auto'.
        """

        self.params.icntl[7] = orderings[ordering]
        t1 = time.clock()
        self.params.job = 1   # analyse
        self.mumps_call()
        t2 = time.clock()

        if self.params.infog[1] < 0:
            raise MUMPSError(self.params.infog[1])

        self.analysed = True

        #self.analysis_stats = AnalysisStatistics(self.params,
        #                                         t2 - t1)

    ####################################################################################################################
    # Solve
    ####################################################################################################################
    cdef solve_dense(self, FLOAT32_t * rhs, INT64_t rhs_length, INT64_t nrhs):
        """
        Solve a linear system after the LU (or LDLt) factorization has previously been performed by `factorize`

        Args:
            rhs: the right hand side (dense matrix or vector)
            rhs_length: Length of each column of the ``rhs``.
            nrhs: Number of columns in the matrix ``rhs``.

        Warning:
            Mumps overwrites ``rhs`` and replaces it by the solution of the linear system.
        """
        #self.params.icntl[9] = 2 if transpose_solve else 1

        self.params.nrhs = <MUMPS_INT> nrhs
        self.params.lrhs = <MUMPS_INT> rhs_length
        self.params.rhs = <FLOAT32_t *>rhs

        self.params.job = 3  # solve
        self.mumps_call()

    cdef solve_sparse(self, INT64_t * rhs_col_ptr, INT64_t * rhs_row_ind,
                       FLOAT32_t * rhs_val, INT64_t rhs_nnz, INT64_t nrhs, FLOAT32_t * x, INT64_t x_length):
        """
        Solve a linear system after the LU (or LDL^t) factorization has previously been performed by `factorize`

        Args:
            rhs_length: Length of each column of the ``rhs``.
            nrhs: Number of columns in the matrix ``rhs``.
            overwrite_rhs : ``True`` or ``False``
                whether the data in ``rhs`` may be overwritten, which can lead to a small
                performance gain. Default is ``False``.
            x : the solution to the linear system as a dense matrix or vector.

        Warning:
            Mumps overwrites ``rhs`` and replaces it by the solution of the linear system.

        """
        #self.params.icntl[9] = 2 if transpose_solve else 1

        self.params.nz_rhs = rhs_nnz
        self.params.nrhs = nrhs # nrhs -1 ?
        self.params.rhs_sparse = <FLOAT32_t *> rhs_val
        self.params.irhs_sparse = <MUMPS_INT *> rhs_row_ind
        self.params.irhs_ptr = <MUMPS_INT *> rhs_col_ptr

        # Mumps places the solution(s) of the linear system in its dense rhs...
        self.params.lrhs = <MUMPS_INT> x_length
        self.params.rhs = <FLOAT32_t *> x

        self.params.job = 3        # solve
        self.params.icntl[20] = 1  # tell solver rhs is sparse
        self.mumps_call()

    def solve(self, **kwargs):
        """

        Args:
            rhs: dense NumPy array (matrix or vector).
            rhs_col_ptr, rhs_row_ind, rhs_val: sparse NumPy CSC arrays (matrix or vector).
            transpose_solve : ``True`` or ``False`` whether to solve A * x = rhs or A^T * x = rhs. Default is ``False``

        Returns:
            Dense NumPy array ``x`` (matrix or vector) with the solution(s) of the linear system.
        """
        if not self.factored:
            self.factorize()

        transpose_solve = kwargs.get('transpose_solve', False)
        self.params.icntl[9] = 2 if transpose_solve else 1

        cdef:
            INT64_t nrhs

        # rhs can be dense or sparse
        if 'rhs' in kwargs:
            rhs = kwargs['rhs']

            if not cnp.PyArray_Check(rhs):
                raise TypeError('rhs dense arrays must be an NumPy array')

            # check is dimensions are OK
            rhs_shape = rhs.shape

            if (rhs_shape[0] != self.nrow):
                raise ValueError("Right hand side has wrong size"
                                 "Attempting to solve the linear system, where A is of size (%d, %d) "
                                 "and rhs is of size (%g)"%(self.nrow, self.nrow, rhs_shape))


            assert are_mixed_types_compatible(FLOAT32_T, rhs.dtype), "Rhs matrix or vector must have a Numpy compatible type (%s)!" % cysparse_to_numpy_type(FLOAT32_T)


            # create x
            x = cnp.asfortranarray(rhs.copy())

            # test number of columns in rhs
            if rhs.ndim == 1:
                nrhs = 1
            else:
                nrhs = <INT64_t> rhs_shape[1]

            self.solve_dense(<FLOAT32_t *> cnp.PyArray_DATA(x), rhs_shape[0], nrhs)

        elif ['rhs_col_ptr', 'rhs_row_ind', 'rhs_val'] in kwargs:
            pass
        else:
            raise TypeError('rhs not given in the right format (dense: rhs=..., sparse: rhs_col_ptr=..., rhs_row_ind=..., rhs_val=...)')



