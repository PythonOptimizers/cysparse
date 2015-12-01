from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython cimport Py_INCREF, Py_DECREF


from cysparse.cysparse_types.cysparse_generic_types cimport split_array_complex_values_kernel_INT64_t_COMPLEX128_t, join_array_complex_values_kernel_INT64_t_COMPLEX128_t


from cysparse.sparse.csc_mat_matrices.csc_mat_INT64_t_COMPLEX128_t cimport CSCSparseMatrix_INT64_t_COMPLEX128_t, MakeCSCSparseMatrix_INT64_t_COMPLEX128_t

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "cholmod.h":

    char * CHOLMOD_DATE
    #ctypedef long SuiteSparse_long # doesn't work... why?
    cdef enum:
        CHOLMOD_MAIN_VERSION
        CHOLMOD_SUB_VERSION
        CHOLMOD_SUBSUB_VERSION
        CHOLMOD_VERSION


    cdef enum:
        # Five objects
        CHOLMOD_COMMON
        CHOLMOD_SPARSE
        CHOLMOD_FACTOR
        CHOLMOD_DENSE
        CHOLMOD_TRIPLET

    # we only use REAL and ZOMPLEX
    cdef enum:
        CHOLMOD_PATTERN  	# pattern only, no numerical values
        CHOLMOD_REAL		# a real matrix
        CHOLMOD_COMPLEX     # a complex matrix (ANSI C99 compatible)
        CHOLMOD_ZOMPLEX     # a complex matrix (MATLAB compatible)

    # itype: we only use INT and LONG
    cdef enum:
        CHOLMOD_INT         # all integer arrays are int
        CHOLMOD_INTLONG     # most are int, some are SuiteSparse_long
        CHOLMOD_LONG        # all integer arrays are SuiteSparse_long

    # dtype: float or double
    cdef enum:
        CHOLMOD_DOUBLE      # all numerical values are double
        CHOLMOD_SINGLE

    cdef enum:
        CHOLMOD_A    		# solve Ax=b
        CHOLMOD_LDLt        # solve LDL'x=b
        CHOLMOD_LD          # solve LDx=b
        CHOLMOD_DLt  	    # solve DL'x=b
        CHOLMOD_L    	    # solve Lx=b
        CHOLMOD_Lt   	    # solve L'x=b
        CHOLMOD_D    	    # solve Dx=b
        CHOLMOD_P    	    # permute x=Px
        CHOLMOD_Pt   	    # permute x=P'x

    int cholmod_l_start(cholmod_common *Common)
    int cholmod_l_finish(cholmod_common *Common)

    int cholmod_l_defaults(cholmod_common *Common)

    # Common struct
    int cholmod_l_check_common(cholmod_common *Common)
    int cholmod_l_print_common(const char *name, cholmod_common *Common)

    # Sparse struct
    int cholmod_l_check_sparse(cholmod_sparse *A, cholmod_common *Common)
    int cholmod_l_print_sparse(cholmod_sparse *A, const char *name, cholmod_common *Common)
    int cholmod_l_free_sparse(cholmod_sparse **A, cholmod_common *Common)

    # Dense struct
    int cholmod_l_free_dense(cholmod_dense **X, cholmod_common *Common)

    # Factor struct
    int cholmod_l_check_factor(cholmod_factor *L, cholmod_common *Common)
    int cholmod_l_print_factor(cholmod_factor *L, const char *name, cholmod_common *Common)
    #int cholmod_l_free_factor()
    # factor_to_sparse

    # Memory management
    void * cholmod_l_free(size_t n, size_t size,	void *p,  cholmod_common *Common)

    # Triplet struct
    #int cholmod_l_check_triplet(cholmod_triplet *T, cholmod_common *Common)
    #print_triplet

    # ANALYZE
    cholmod_factor * cholmod_l_analyze(cholmod_sparse *A,cholmod_common *Common)
    int  cholmod_l_check_factor(cholmod_factor *L, cholmod_common *Common)
    
    # FACTORIZE
    int cholmod_l_factorize(cholmod_sparse *, cholmod_factor *, cholmod_common *)
    int cholmod_l_free_factor(cholmod_factor **LHandle, cholmod_common *Common)

    # SOLVE
    cholmod_dense * cholmod_l_solve (int, cholmod_factor *, cholmod_dense *, cholmod_common *)
    cholmod_sparse * cholmod_l_spsolve (int, cholmod_factor *, cholmod_sparse *,
    cholmod_common *)

CHOLMOD_SYS_DICT = {
        'CHOLMOD_A'     : CHOLMOD_A
    }

########################################################################################################################
# CHOLMOD HELPERS
########################################################################################################################
##################################################################
# FROM CSCSparseMatrix -> cholmod_sparse
##################################################################
# Populating a sparse matrix in CHOLMOD is done in two steps:
# - first (populate1), we give the common attributes and
# - second (populate2), we split the values array in two if needed (complex case) and give the values (real or complex).

cdef populate1_cholmod_sparse_struct_with_CSCSparseMatrix(cholmod_sparse * sparse_struct, CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat, bint no_copy=True):
    """
    Populate a CHOLMO C struct ``cholmod_sparse`` with the content of a :class:`CSCSparseMatrix_INT64_t_COMPLEX128_t` matrix.

    First part: common attributes for both real and complex matrices.

    Note:
        We only use the ``cholmod_sparse`` **packed** and **sorted** version.
    """
    assert no_copy, "The version with copy is not implemented yet..."

    assert(csc_mat.are_row_indices_sorted()), "We only use CSC matrices with internal row indices sorted. The non sorted version is not implemented yet."

    sparse_struct.nrow = csc_mat.nrow
    sparse_struct.ncol = csc_mat.ncol
    sparse_struct.nzmax = csc_mat.nnz

    sparse_struct.p = csc_mat.ind
    sparse_struct.i = csc_mat.row

    # TODO: change this when we'll accept symmetric matrices **without** symmetric storage scheme
    if csc_mat.is_symmetric:
        sparse_struct.stype = -1
    else:
        sparse_struct.stype = 0

    # itype: can be CHOLMOD_INT or CHOLMOD_LONG: we don't use the mixed version CHOLMOD_INTLONG

    sparse_struct.itype = CHOLMOD_LONG


    sparse_struct.sorted = 1                                 # TRUE if columns are sorted, FALSE otherwise
    sparse_struct.packed = 1                                 # We use the packed CSC version: **no** need to construct
                                                             # the nz (array with number of non zeros by column)



cdef populate2_cholmod_sparse_struct_with_CSCSparseMatrix(cholmod_sparse * sparse_struct,
                                                              CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat,
                                                              FLOAT64_t * csc_mat_rval,
                                                              FLOAT64_t * csc_mat_ival,
                                                              bint no_copy=True):
    """
    Populate a CHOLMO C struct ``cholmod_sparse`` with the content of a :class:`CSCSparseMatrix_INT64_t_COMPLEX128_t` matrix.

    Second part: Non common attributes for complex matrices.

    Note:
        We only use the ``cholmod_sparse`` **packed** version.
    """
    assert no_copy, "The version with copy is not implemented yet..."


    sparse_struct.x = csc_mat_rval
    sparse_struct.z = csc_mat_ival

    sparse_struct.xtype = CHOLMOD_ZOMPLEX                    # CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX
    sparse_struct.dtype = CHOLMOD_DOUBLE




##################################################################
# FROM cholmod_sparse -> CSCSparseMatrix
##################################################################
cdef CSCSparseMatrix_INT64_t_COMPLEX128_t cholmod_sparse_to_CSCSparseMatrix_INT64_t_COMPLEX128_t(cholmod_sparse * sparse_struct, bint no_copy=False):
    """
    Convert a ``cholmod`` sparse struct to a :class:`CSCSparseMatrix_INT64_t_COMPLEX128_t`.

    """
    # TODO: generalize to any cholmod sparse structure, with or without copy
    # TODO: generalize to complex case
    # TODO: remove asserts
    assert sparse_struct.sorted == 1, "We only accept cholmod_sparse matrices with sorted indices"
    assert sparse_struct.packed == 1, "We only accept cholmod_sparse matrices with packed indices"

    assert sparse_struct.xtype == CHOLMOD_ZOMPLEX, "We only accept cholmod_sparse matrices with zomplex"


    cdef:
        CSCSparseMatrix_INT64_t_COMPLEX128_t csc_mat
        INT64_t nrow
        INT64_t ncol
        INT64_t nnz
        bint is_symmetric = False

        # internal arrays of the CSC matrix
        INT64_t * ind
        INT64_t * row

        # internal arrays of the cholmod sparse matrix
        INT64_t * ind_cholmod
        INT64_t * row_cholmod

        # internal arrays for the CSC matrix
        FLOAT64_t * valx
        FLOAT64_t * valz

        COMPLEX128_t * val_complex

        # internal arrays for the cholmod sparse matrix
        FLOAT64_t * valx_cholmod
        FLOAT64_t * valz_cholmod



        INT64_t j, k

    nrow = sparse_struct.nrow
    ncol = sparse_struct.ncol
    nnz = sparse_struct.nzmax

    if sparse_struct.stype == 0:
        is_symmetric = False
    elif sparse_struct.stype < 0:
        is_symmetric == True
    else:
        raise NotImplementedError('We do not accept cholmod square symmetric sparse matrix with upper triangular part filled in.')

    ##################################### NO COPY ######################################################################
    if no_copy:
        ind = <INT64_t *> sparse_struct.p
        row = <INT64_t *> sparse_struct.i

        valx = <FLOAT64_t *> sparse_struct.x
        valz = <FLOAT64_t *> sparse_struct.z

    ##################################### WITH COPY ####################################################################
    else:   # we do a copy

        ind_cholmod = <INT64_t * > sparse_struct.p
        row_cholmod = <INT64_t * > sparse_struct.i

        ind = <INT64_t *> PyMem_Malloc((ncol + 1) * sizeof(INT64_t))

        if not ind:
            raise MemoryError()

        row = <INT64_t *> PyMem_Malloc(nnz * sizeof(INT64_t))

        if not row:
            PyMem_Free(ind)
            PyMem_Free(row)

            raise MemoryError()


        for j from 0 <= j <= ncol:
            ind[j] = ind_cholmod[j]

        for k from 0 <= k < nnz:
            row[k] = row_cholmod[k]



        valx_cholmod = <FLOAT64_t *> sparse_struct.x
        valz_cholmod = <FLOAT64_t *> sparse_struct.z

        valx = <FLOAT64_t *> PyMem_Malloc(nnz * sizeof(FLOAT64_t))

        if not valx:
            PyMem_Free(ind)
            PyMem_Free(row)
            PyMem_Free(valx)

            raise MemoryError()

        valz = <FLOAT64_t *> PyMem_Malloc(nnz * sizeof(FLOAT64_t))

        if not valz:
            PyMem_Free(ind)
            PyMem_Free(row)
            PyMem_Free(valx)
            PyMem_Free(valz)

            raise MemoryError()

        for k from 0 <= k < nnz:
            valx[k] = valx_cholmod[k]
            valz[k] = valz_cholmod[k]



    raise NotImplementedError('Complex case not implemented yet...')


    return csc_mat


##################################################################
# FROM NumPy ndarray -> cholmod_dense
##################################################################
cdef cholmod_dense numpy_ndarray_to_cholmod_dense(cnp.ndarray[cnp.npy_complex128, ndim=1, mode="c"] b):
    """
    Convert a :program:`NumPy` one dimensionnal array to the corresponding ``cholmod_dense`` matrix.
    """
    # access b
    cdef COMPLEX128_t * b_data = <COMPLEX128_t *> cnp.PyArray_DATA(b)

    # Creation of CHOLMOD DENSE MATRIX
    cdef cholmod_dense B
    B = cholmod_dense()

    B.nrow = b.shape[0]
    B.ncol = 1

    B.nzmax = b.shape[0]

    B.d = b.shape[0]


    # TODO: to be done!
    raise NotImplementedError("Not yet...")

    B.xtype = CHOLMOD_ZOMPLEX                    # CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX
    B.dtype = CHOLMOD_DOUBLE


    return B

##################################################################
# FROM cholmod_dense -> NumPy ndarray
##################################################################
cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode="c"] cholmod_dense_to_numpy_ndarray(cholmod_dense * b):
    raise NotImplementedError()

########################################################################################################################
# CHOLMOD
########################################################################################################################

def cholmod_version():
    version_string = "CHOLMOD version %s" % CHOLMOD_VERSION

    return version_string

def cholmod_detailed_version():
    version_string = "%s.%s.%s (%s)" % (CHOLMOD_MAIN_VERSION,
                                         CHOLMOD_SUB_VERSION,
                                         CHOLMOD_SUBSUB_VERSION,
                                         CHOLMOD_DATE)
    return version_string

cdef class CholmodContext_INT64_t_COMPLEX128_t:
    """
    Cholmod Context from SuiteSparse.

    This version **only** deals with ``LLSparseMatrix_INT64_t_COMPLEX128_t`` objects.

    We follow the common use of Cholmod. In particular, we use the same names for the methods of this
    class as their corresponding counter-parts in Cholmod.
    """
    CHOLMOD_VERSION = "%s.%s.%s (%s)" % (CHOLMOD_MAIN_VERSION,
                                     CHOLMOD_SUB_VERSION,
                                     CHOLMOD_SUBSUB_VERSION,
                                     CHOLMOD_DATE)

    ####################################################################################################################
    # INIT
    ####################################################################################################################
    def __cinit__(self, LLSparseMatrix_INT64_t_COMPLEX128_t A):
        """
        """
        self.A = A
        Py_INCREF(self.A)  # increase ref to object to avoid the user deleting it explicitly or implicitly

        self.nrow = A.nrow
        self.ncol = A.ncol

        self.nnz = self.A.nnz

        # test if we can use CHOLMOD
        assert self.nrow == self.ncol, "Only square matrices are handled in CHOLMOD"
        # TODO: change this. This is an assumption that is too strong
        assert self.A.is_symmetric, "Only symmetric matrices (using the symmetric storage scheme) are handled in CHOLMOD"

        self.csc_mat = self.A.to_csc()

        # CHOLMOD
        self.common_struct = cholmod_common()
        cholmod_l_start(&self.common_struct)

        self.sparse_struct = cholmod_sparse()
        # common attributes for real and complex matrices
        populate1_cholmod_sparse_struct_with_CSCSparseMatrix(&self.sparse_struct, self.csc_mat)


        cdef:
            FLOAT64_t * rval
            FLOAT64_t * ival

        rval = <FLOAT64_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT64_t))
        if not rval:
            raise MemoryError()
        self.csc_rval = rval

        ival = <FLOAT64_t *> PyMem_Malloc(self.nnz * sizeof(FLOAT64_t))
        if not ival:
            PyMem_Free(rval)
            raise MemoryError()
        self.csc_ival = ival

        # split array of complex values into two real value arrays
        split_array_complex_values_kernel_INT64_t_COMPLEX128_t(self.csc_mat.val, self.nnz,
                                                     self.csc_rval, self.nnz,
                                                     self.csc_ival, self.nnz)



        self.factor_struct_initialized = False
        self.already_factorized = False


    ####################################################################################################################
    # Properties
    ####################################################################################################################
    # Propreties that bear the same name as a reserved Python keyword, are prefixed by 'c_'.
    ######################################### COMMON STRUCT Properties #################################################
    # Printing
    property c_print:
        def __get__(self): return self.common_struct.print_
        def __set__(self, value): self.common_struct.print_ = value

    property precise:
        def __get__(self): return self.common_struct.precise
        def __set__(self, value): self.common_struct.precise = value

    property try_catch:
        def __get__(self): return self.common_struct.try_catch
        def __set__(self, value): self.common_struct.try_catch = value

    ####################################################################################################################
    # FREE MEMORY
    ####################################################################################################################
    def __dealloc__(self):
        """

        """
        # we don't delete sparse_struct as **all** arrays are allocated in self.csc_mat
        # TODO: doesn't work... WHY?
        #del self.csc_mat

        if self.factor_struct_initialized:
            cholmod_l_free_factor(&self.factor_struct, &self.common_struct)

        cholmod_l_finish(&self.common_struct)

        Py_DECREF(self.A) # release ref

    ####################################################################################################################
    # COMMON OPERATIONS
    ####################################################################################################################
    def reset_default_parameters(self):
        cholmod_l_defaults(&self.common_struct)

    cpdef bint check_matrix(self):
        """
        Check if internal CSC matrix is OK.

        Returns:
            ``True`` if everything is OK, ``False`` otherwise. Depending on the verbosity, some error messages can
            be displayed on ``sys.stdout``.
        """
        return cholmod_l_check_sparse(&self.sparse_struct, &self.common_struct)

    def analyze(self):
        if not self.factor_struct_initialized:
            self.factor_struct = <cholmod_factor *> cholmod_l_analyze(&self.sparse_struct,&self.common_struct)
            self.factor_struct_initialized = True

    cpdef bint check_factor(self):
        if self.factor_struct_initialized:
            return cholmod_l_check_factor(self.factor_struct, &self.common_struct)

        return False

    def factorize(self, force = False):
        # if needed
        self.analyze()

        if not self.already_factorized or force:
            cholmod_l_factorize(&self.sparse_struct, self.factor_struct, &self.common_struct)
            self.already_factorized = True

    def solve(self, cnp.ndarray[cnp.npy_complex128, ndim=1, mode="c"] b, cholmod_sys='CHOLMOD_A'):

        # test argument b
        cdef cnp.npy_intp * shape_b
        try:
            shape_b = b.shape
        except:
            raise AttributeError("argument b must implement attribute 'shape'")

        dim_b = shape_b[0]
        assert dim_b == self.nrow, "array dimensions must agree"

        if cholmod_sys not in CHOLMOD_SYS_DICT.keys():
            raise ValueError('cholmod_sys must be in' % CHOLMOD_SYS_DICT.keys())

        # if needed
        self.factorize()

        # convert NumPy array to CHOLMOD dense vector
        cdef cholmod_dense B

        B = numpy_ndarray_to_cholmod_dense(b)

        cdef cholmod_dense * cholmod_sol
        cholmod_sol = cholmod_l_solve(CHOLMOD_SYS_DICT[cholmod_sys], self.factor_struct, &B, &self.common_struct)

        # TODO: free B
        # TODO: convert sol to NumPy array

        cdef cnp.ndarray[cnp.npy_complex128, ndim=1, mode='c'] sol = np.empty(self.ncol, dtype=np.complex128)

        # make a copy
        cdef INT64_t j

        cdef COMPLEX128_t * cholmod_sol_array_ptr = <COMPLEX128_t * > cholmod_sol.x


        raise NotImplementedError("To be coded")


        # Free CHOLMOD dense solution
        cholmod_l_free_dense(&cholmod_sol, &self.common_struct)

        return sol

    ####################################################################################################################
    # GPU
    ####################################################################################################################
    def request_GPU(self):
        """
        GPU-acceleration is requested.

        If GPU processing is requested but there is no GPU present, CHOLMOD will continue using the CPU only.
        Consequently it is **always safe** to request GPU processing.

        """
        self.common_struct.useGPU = 1

    def prohibit_GPU(self):
        """
        GPU-acceleration is explicitely prohibited.

        """
        self.common_struct.useGPU = 0

    ####################################################################################################################
    # PRINTING
    ####################################################################################################################
    def print_common_struct(self):
        cholmod_l_print_common("cholmod_common_struct", &self.common_struct)

    def print_sparse_matrix(self):
        cholmod_l_print_sparse(&self.sparse_struct, "cholmod_sparse_matrix", &self.common_struct)