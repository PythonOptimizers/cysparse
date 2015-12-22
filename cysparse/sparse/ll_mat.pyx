
from cysparse.sparse.ll_mat cimport LL_MAT_DEFAULT_SIZE_HINT
from cysparse.sparse.s_mat cimport unexposed_value

from cysparse.common_types.cysparse_types import *
from cysparse.common_types.cysparse_types cimport *
from cysparse.common_types.cysparse_types cimport min_integer_type
from cysparse.common_types.cysparse_numpy_types import are_mixed_types_cast_compatible

#from cysparse.common_types.cysparse_generic_types cimport min_type

from cysparse.sparse.s_mat cimport SparseMatrix

from cython cimport isinstance
from libc.stdio cimport *
from libc.string cimport *
from cpython.unicode cimport *
from libc.stdlib cimport *

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef extern from "Python.h":
    # *** Types ***
    Py_ssize_t PY_SSIZE_T_MAX
    int PyInt_Check(PyObject *o)
    long PyInt_AS_LONG(PyObject *io)

    # *** Slices ***
    ctypedef struct PySliceObject:
        pass

    # Cython's version doesn't work for all versions...
    int PySlice_GetIndicesEx(
        PySliceObject* s, Py_ssize_t length,
        Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step,
        Py_ssize_t *slicelength) except -1

    int PySlice_Check(PyObject *ob)

    # *** List ***
    int PyList_Check(PyObject *p)
    PyObject* PyList_GetItem(PyObject *list, Py_ssize_t index)
    Py_ssize_t PyList_Size(PyObject *list)

    PyObject* Py_BuildValue(char *format, ...)
    PyObject* PyList_New(Py_ssize_t len)
    void PyList_SET_ITEM(PyObject *list, Py_ssize_t i, PyObject *o)
    PyObject* PyList_GET_ITEM(PyObject *list, Py_ssize_t i)

LL_MAT_INCREASE_FACTOR = 1.5
LL_MAT_DEFAULT_SIZE_HINT = 40

LL_MAT_PPRINT_COL_THRESH = 20
LL_MAT_PPRINT_ROW_THRESH = 40

########################################################################################################################
# Cython, NumPy import/cimport
########################################################################################################################
# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

cnp.import_array()

########################################################################################################################
# CySparse include
########################################################################################################################
# pxi files should come last (except for circular dependencies)

    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT32_t cimport LLSparseMatrix_INT32_t_INT32_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT64_t cimport LLSparseMatrix_INT32_t_INT64_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT32_t cimport LLSparseMatrix_INT32_t_FLOAT32_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT64_t cimport LLSparseMatrix_INT32_t_FLOAT64_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT128_t cimport LLSparseMatrix_INT32_t_FLOAT128_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX64_t cimport LLSparseMatrix_INT32_t_COMPLEX64_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX128_t cimport LLSparseMatrix_INT32_t_COMPLEX128_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_COMPLEX256_t cimport LLSparseMatrix_INT32_t_COMPLEX256_t
    

    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_INT32_t cimport LLSparseMatrix_INT64_t_INT32_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_INT64_t cimport LLSparseMatrix_INT64_t_INT64_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT32_t cimport LLSparseMatrix_INT64_t_FLOAT32_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT64_t cimport LLSparseMatrix_INT64_t_FLOAT64_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT128_t cimport LLSparseMatrix_INT64_t_FLOAT128_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX64_t cimport LLSparseMatrix_INT64_t_COMPLEX64_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX128_t cimport LLSparseMatrix_INT64_t_COMPLEX128_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_COMPLEX256_t cimport LLSparseMatrix_INT64_t_COMPLEX256_t
    




    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT32_t_INT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT32_t_INT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT32_t_INT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT32_t_INT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT32_t_INT32_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT32_t_INT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT32_t_INT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT32_t_INT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT32_t_INT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT32_t_INT64_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT32_t_FLOAT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT32_t_FLOAT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT32_t_FLOAT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT32_t_FLOAT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT32_t_FLOAT32_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT32_t_FLOAT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT32_t_FLOAT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT32_t_FLOAT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT32_t_FLOAT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT32_t_FLOAT64_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT32_t_FLOAT128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT32_t_FLOAT128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT32_t_FLOAT128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT32_t_FLOAT128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT32_t_FLOAT128_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT32_t_COMPLEX64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT32_t_COMPLEX64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT32_t_COMPLEX64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT32_t_COMPLEX64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT32_t_COMPLEX64_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT32_t_COMPLEX128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT32_t_COMPLEX128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT32_t_COMPLEX128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT32_t_COMPLEX128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT32_t_COMPLEX128_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT32_t_COMPLEX256_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT32_t_COMPLEX256_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT32_t_COMPLEX256_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT32_t_COMPLEX256_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT32_t_COMPLEX256_t.pxi"
    

    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT64_t_INT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT64_t_INT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT64_t_INT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT64_t_INT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT64_t_INT32_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT64_t_INT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT64_t_INT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT64_t_INT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT64_t_INT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT64_t_INT64_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT64_t_FLOAT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT64_t_FLOAT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT64_t_FLOAT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT64_t_FLOAT32_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT64_t_FLOAT32_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT64_t_FLOAT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT64_t_FLOAT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT64_t_FLOAT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT64_t_FLOAT64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT64_t_FLOAT64_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT64_t_FLOAT128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT64_t_FLOAT128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT64_t_FLOAT128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT64_t_FLOAT128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT64_t_FLOAT128_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT64_t_COMPLEX64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT64_t_COMPLEX64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT64_t_COMPLEX64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT64_t_COMPLEX64_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT64_t_COMPLEX64_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT64_t_COMPLEX128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT64_t_COMPLEX128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT64_t_COMPLEX128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT64_t_COMPLEX128_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT64_t_COMPLEX128_t.pxi"
    
include "ll_mat_matrices/ll_mat_constructors/ll_mat_arrowheads_INT64_t_COMPLEX256_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_diagonals_INT64_t_COMPLEX256_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_bands_INT64_t_COMPLEX256_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_linear_fills_INT64_t_COMPLEX256_t.pxi"
include "ll_mat_matrices/ll_mat_constructors/ll_mat_permutations_INT64_t_COMPLEX256_t.pxi"
    


########################################################################################################################
# Matrix Market
########################################################################################################################
cdef:
    str MM_MATRIX_MARKET_BANNER_STR = "%%MatrixMarket"
    str MM_MTX_STR = "matrix"
    str MM_ARRAY_STR = "array"
    str MM_COORDINATE_STR = "coordinate"
    str MM_COMPLEX_STR = "complex"
    str MM_REAL_STR = "real"
    str MM_INT_STR = "integer"
    str MM_PATTERN_STR = "pattern"
    str MM_GENERAL_STR = "general"
    str MM_SYMM_STR = "symmetric"
    str MM_HERM_STR = "hermitian"
    str MM_SKEW_STR = "skew"


cdef enum:
    MM_COMPLEX  = 0
    MM_REAL     = 1
    MM_INTEGER  = 2
    MM_PATTERN  = 3

cdef enum:
    MM_GENERAL   = 11
    MM_HERMITIAN = 12
    MM_SYMMETRIC = 13
    MM_SKEW      = 14

def get_mm_matrix_dimension_specifications(mm_filename):
    """
    Return n, m, nnz from a Matrix Market file.

    Note:
        This is done very "stupidly": we seek the first line of real data (blank lines and comment lines are disregarded)
        and try to read ``n, m, nnz``.

    Raises:
        ``IOError``: if we cannot read ``n, m, nnz`` on the first line of real data.
    """
    with open(mm_filename, 'r') as f:
        for line in f:
            # avoid empty lines
            if line.rstrip():
                # avoid comment lines
                if not line.startswith('%'):
                    # first line that is not an empty or comment line
                    tokens = line.split()
                    if len(tokens) != 3:
                        raise IOError('Matrix Market file not recognized: first data line should contain n, m, nnz')
                    else:
                        return long(tokens[0]), long(tokens[1]), float(tokens[2])

    raise IOError('Matrix Market file not recognized: first data line should contain n, m, nnz')

def get_mm_matrix_type_specifications(mm_filename):
    """
    Return 4-tuple (strings) given in the Matrix Market banner.

    Raises:
        ``IOError`` if first line doesn't contain the banner or if the banner is not recognized.
    """
    with open(mm_filename, 'r') as f:
        # read banner
        line = f.readline()
        token_list = line.split()

        if len(token_list) != 5:
            raise IOError('Matrix format not recognized as Matrix Market format: not the right number of tokens in the Matrix Market banner')

        # MATRIX MARKET BANNER START
        if token_list[0] != MM_MATRIX_MARKET_BANNER_STR:
            raise IOError('Matrix format not recognized as Matrix Market format: zeroth token in the Matrix Market banner is not "%s"' % MM_MATRIX_MARKET_BANNER_STR)

    return token_list[1:]


    
include "ll_mat_matrices/ll_mat_IO/ll_mat_mm_INT32_t_INT64_t.pxi"
    
include "ll_mat_matrices/ll_mat_IO/ll_mat_mm_INT32_t_FLOAT64_t.pxi"
    
include "ll_mat_matrices/ll_mat_IO/ll_mat_mm_INT32_t_COMPLEX128_t.pxi"
    

    
include "ll_mat_matrices/ll_mat_IO/ll_mat_mm_INT64_t_INT64_t.pxi"
    
include "ll_mat_matrices/ll_mat_IO/ll_mat_mm_INT64_t_FLOAT64_t.pxi"
    
include "ll_mat_matrices/ll_mat_IO/ll_mat_mm_INT64_t_COMPLEX128_t.pxi"
    



########################################################################################################################
# Common matrix operations
########################################################################################################################
cpdef bint PyLLSparseMatrix_Check(object obj):
    """
    Test if ``obj`` is a :class:`LLSparseMatrix`.

    """
    cdef:
        bint is_ll_sparse_matrix = False

    if isinstance(obj, SparseMatrix):
        is_ll_sparse_matrix = obj.type == 'LLSparseMatrix'

    return is_ll_sparse_matrix

def matvec(A, b):
    """
    Return :math:`A * b`.
    """
    # TODO: test input arguments?
    return A.matvec(b)

def matvec_transp(A, b):
    """
    Return :math:`A^t*b`.
    """
    # TODO: test input arguments?
    return A.matvec_transp(b)

########################################################################################################################
# General factory methods
########################################################################################################################
def LLSparseMatrix(**kwargs):
    """
    Main factory method to create a (filled or empty) ``LLSparseMatrix`` matrix.

    An ``LLSparseMatrix`` can be created in three different ways (cases):
        - from specifications;
        - from another matrix;
        - from a file.

    In the first case, from specifications, an empty matrix is created. To distinguish between these 3 cases, use the
    right named arguments.

    Warning:
        **All** arguments are named.

    In all cases, you **can** supply an ``itype`` (index type) and a ``dtype`` (element type). By default (i.e. if you
    don't provide these arguments, ``itype == INT32_T`` and ``dtype == FLOAT64_T``) and specify a *storage format*:
        - ``store_zero``: if ``True``, store explicitely zeros (default: ``False``);
        - ``store_symmetric``: if ``True``, use symmetric storage, i.e. only the lower triangular part of the matrix is stored (by default: ``False``);

    If a matrix or filename is supplied, these arguments **must** coincide with the supplied matrix types. If not, an error is thrown.

    To create an ``LLSparseMatrix`` matrix, use the following arguments:

    * Case 1: From specifications. Use ``nrow`` and ``ncol`` or ``size`` to specify the dimension of the new matrix. You can provide a
        ``size_hint`` argument to (pre)allocate some space for the elements in advance.
    * Case 2: From another matrix. Use the ``matrix`` argument.
        This is not yet implemented.
    * Case 3: From a file.
        By default, a ``test_bounds`` argument is set to ``True`` to test if all indices are not out of bounds. You can disable this to gain
        some speed when reading a file.
        For the moment, only the Matrix Market format is available. See :func:`LLSparseMatrixFromMMFile` if you have no idea of the
        matrix type contained in a matrix market file. To give a file name of a file in Matrix Market format, use the ``mm_filename`` argument.

    """
    ####################################################################################################################
    #                                            *** Get arguments ***
    ####################################################################################################################
    # General specifications
    nrow = kwargs.get('nrow', -1)
    ncol = kwargs.get('ncol', -1)
    size = kwargs.get('size', -1)
    size_hint = kwargs.get('size_hint', LL_MAT_DEFAULT_SIZE_HINT)

    #itype = kwargs.get('itype', INT32_T)
    itype = kwargs.get('itype', INT64_T)
    dtype = kwargs.get('dtype', FLOAT64_T)

    assert itype in INDEX_TYPES, "itype not recognized"
    assert dtype in ELEMENT_TYPES, "dtype not recognized"

    assert itype in [INT32_T,INT64_T], "itype is not accepted as index type"
    assert dtype in [INT32_T,INT64_T,FLOAT32_T,FLOAT64_T,FLOAT128_T,COMPLEX64_T,COMPLEX128_T,COMPLEX256_T], "dtype is not accepted as type for a matrix element"

    cdef bint store_zero = kwargs.get('store_zero', False)
    cdef bint store_symmetric = kwargs.get('store_symmetric', False)
    cdef bint test_bounds = kwargs.get('test_bounds', True)

    # From matrices
    matrix = kwargs.get('matrix', None)

    # From file names
    from_filename = False
    mm_filename = kwargs.get('mm_filename', None)
    if mm_filename is not None:
        from_filename = True

    if matrix is not None or from_filename:
        assert (matrix is not None) != (from_filename), "Cannot use a matrix and a file to create a LLSparseMatrix"

    mm_read_file_experimental = kwargs.get('mm_experimental', None) is not None

    if mm_read_file_experimental:
        print "Try experimental reading of MM files"

    ####################################################################################################################
    #                                            *** Case dispatch ***
    ####################################################################################################################
    real_nrow = -1
    real_ncol = -1

    #                                            CASE 1: from specifications
    if matrix is None and mm_filename is None:
        if nrow != -1 and ncol != -1:
            if size != -1:
                assert nrow == ncol == size, "Mismatch between nrow, ncol and size"
            real_nrow = nrow
            real_ncol = ncol
        elif nrow != -1 and ncol == -1:
            if size != -1:
                assert size == nrow, "Mismatch between nrow and size"
            real_nrow = nrow
            real_ncol = nrow
        elif nrow == -1 and ncol != -1:
            if size != -1:
                assert ncol == size, "Mismatch between ncol and size"
            real_nrow = ncol
            real_ncol = ncol
        else:
            assert size != -1, "No size given"
            real_nrow = size
            real_ncol = size


    
        if itype == INT32_T:
    
    
        
            if dtype == INT32_T:
        
                return LLSparseMatrix_INT32_t_INT32_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == INT64_T:
        
                return LLSparseMatrix_INT32_t_INT64_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == FLOAT32_T:
        
                return LLSparseMatrix_INT32_t_FLOAT32_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == FLOAT64_T:
        
                return LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == FLOAT128_T:
        
                return LLSparseMatrix_INT32_t_FLOAT128_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == COMPLEX64_T:
        
                return LLSparseMatrix_INT32_t_COMPLEX64_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == COMPLEX128_T:
        
                return LLSparseMatrix_INT32_t_COMPLEX128_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == COMPLEX256_T:
        
                return LLSparseMatrix_INT32_t_COMPLEX256_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    

    
        elif itype == INT64_T:
    
    
        
            if dtype == INT32_T:
        
                return LLSparseMatrix_INT64_t_INT32_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == INT64_T:
        
                return LLSparseMatrix_INT64_t_INT64_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == FLOAT32_T:
        
                return LLSparseMatrix_INT64_t_FLOAT32_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == FLOAT64_T:
        
                return LLSparseMatrix_INT64_t_FLOAT64_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == FLOAT128_T:
        
                return LLSparseMatrix_INT64_t_FLOAT128_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == COMPLEX64_T:
        
                return LLSparseMatrix_INT64_t_COMPLEX64_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == COMPLEX128_T:
        
                return LLSparseMatrix_INT64_t_COMPLEX128_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    
        
            elif dtype == COMPLEX256_T:
        
                return LLSparseMatrix_INT64_t_COMPLEX256_t(control_object=unexposed_value,
                                                                  nrow=real_nrow,
                                                                  ncol=real_ncol,
                                                                  dtype=dtype,
                                                                  itype=itype,
                                                                  size_hint=size_hint,
                                                                  store_zero=store_zero,
                                                                  store_symmetric=store_symmetric)
    


    #                                            CASE 2: from another matrix
    if matrix is not None:
        raise NotImplementedError("Cannot create a LLSparseMatrix from another matrix (yet)")

    #                                            CASE 3: from a file
    if from_filename:
        if mm_filename is not None:
            # Matrix Market format

            assert itype in [INT32_T,INT64_T], "itype is not accepted as index type for a matrix from a Matrix Market file.\n\Accepted itypes:\n\t" + \
             "INT32_T,INT64_T"

            assert dtype in [INT64_T,FLOAT64_T,COMPLEX128_T], "dtype is not accepted as type for a matrix from a Matrix Market file.\nAccepted dtypes:\n\t" + \
             "INT64_T,FLOAT64_T,COMPLEX128_T"


    
            if itype == INT32_T:
    
    
        
                if dtype == INT64_T:
        
                    if mm_read_file_experimental:
                        return MakeLLSparseMatrixFromMMFile2_INT32_t_INT64_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
                    return MakeLLSparseMatrixFromMMFile_INT32_t_INT64_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
    
        
                elif dtype == FLOAT64_T:
        
                    if mm_read_file_experimental:
                        return MakeLLSparseMatrixFromMMFile2_INT32_t_FLOAT64_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
                    return MakeLLSparseMatrixFromMMFile_INT32_t_FLOAT64_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
    
        
                elif dtype == COMPLEX128_T:
        
                    if mm_read_file_experimental:
                        return MakeLLSparseMatrixFromMMFile2_INT32_t_COMPLEX128_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
                    return MakeLLSparseMatrixFromMMFile_INT32_t_COMPLEX128_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
    

    
            elif itype == INT64_T:
    
    
        
                if dtype == INT64_T:
        
                    if mm_read_file_experimental:
                        return MakeLLSparseMatrixFromMMFile2_INT64_t_INT64_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
                    return MakeLLSparseMatrixFromMMFile_INT64_t_INT64_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
    
        
                elif dtype == FLOAT64_T:
        
                    if mm_read_file_experimental:
                        return MakeLLSparseMatrixFromMMFile2_INT64_t_FLOAT64_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
                    return MakeLLSparseMatrixFromMMFile_INT64_t_FLOAT64_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
    
        
                elif dtype == COMPLEX128_T:
        
                    if mm_read_file_experimental:
                        return MakeLLSparseMatrixFromMMFile2_INT64_t_COMPLEX128_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
                    return MakeLLSparseMatrixFromMMFile_INT64_t_COMPLEX128_t(mm_filename=mm_filename, store_zero=store_zero, test_bounds=test_bounds)
    



def LLSparseMatrixFromMMFile(filename, store_zero=False, test_bounds=True):
    """
    Factory method to create an ``LLSparseMatrix`` matrix from a ``Matrix Market`` file.

    Return the minimal ``LLSparseMatrix`` possible to hold the matrix.

    Raises:
        ``TypeError`` whenever the types for indices and elements of the matrix can not be recognized.
    """

    # Get matrix information
    matrix_object, matrix_type, data_type, storage_format = get_mm_matrix_type_specifications(filename)
    n, m, nnz = get_mm_matrix_dimension_specifications(filename)

    # Define itype
    cdef CySparseType n_type = min_integer_type(n,[INT32_T,INT64_T])

    cdef CySparseType m_type = min_integer_type(m,[INT32_T,INT64_T])

    cdef CySparseType itype = <CySparseType> result_type(n_type, m_type)

    #Define dtype
    cdef CySparseType dtype
    if data_type == MM_COMPLEX_STR:
        dtype = COMPLEX128_T
    elif data_type == MM_REAL_STR:
        dtype = FLOAT64_T
    elif data_type == MM_INT_STR:
        dtype = INT64_T
    else:
        raise TypeError('Element type of matrix is not recognized')

    # launch right factory method

    
    if itype == INT32_T:
    
        
        if dtype == INT64_T:
        
            return MakeLLSparseMatrixFromMMFile_INT32_t_INT64_t(mm_filename=filename, store_zero=store_zero, test_bounds=test_bounds)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeLLSparseMatrixFromMMFile_INT32_t_FLOAT64_t(mm_filename=filename, store_zero=store_zero, test_bounds=test_bounds)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeLLSparseMatrixFromMMFile_INT32_t_COMPLEX128_t(mm_filename=filename, store_zero=store_zero, test_bounds=test_bounds)
    
    

    
    elif itype == INT64_T:
    
        
        if dtype == INT64_T:
        
            return MakeLLSparseMatrixFromMMFile_INT64_t_INT64_t(mm_filename=filename, store_zero=store_zero, test_bounds=test_bounds)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeLLSparseMatrixFromMMFile_INT64_t_FLOAT64_t(mm_filename=filename, store_zero=store_zero, test_bounds=test_bounds)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeLLSparseMatrixFromMMFile_INT64_t_COMPLEX128_t(mm_filename=filename, store_zero=store_zero, test_bounds=test_bounds)
    
    

    else:
        raise TypeError('itype not recognized')


########################################################################################################################
# Special factory methods
########################################################################################################################
def ArrowheadLLSparseMatrix(**kwargs):
    """
    See ``MakeArrowHeadLLSparseMatrix``.

    Note:
        Input arguments are **not** tested.
    """
    element = kwargs.pop('element', None)

    if kwargs.get('store_symmetric', False):
        raise NotImplementedError('This type of matrix is not implemented for symmetric matrices')

    ll_mat = LLSparseMatrix(**kwargs)

    itype = ll_mat.itype
    dtype = ll_mat.dtype

    # create 1.0 element if needed
    if element is None:
        if is_integer_type(dtype):
            element = 1
        elif is_real_type(dtype):
            element = 1.0
        elif is_complex_type(dtype):
            element = 1.0 + 0.0j
        else:
            raise TypeError('dtype not recognized')

    # launch right "constructor" method

    
    if itype == INT32_T:
    
        
        if dtype == INT32_T:
        
            return MakeArrowHeadLLSparseMatrix_INT32_t_INT32_t(ll_mat, element)
    
        
        elif dtype == INT64_T:
        
            return MakeArrowHeadLLSparseMatrix_INT32_t_INT64_t(ll_mat, element)
    
        
        elif dtype == FLOAT32_T:
        
            return MakeArrowHeadLLSparseMatrix_INT32_t_FLOAT32_t(ll_mat, element)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeArrowHeadLLSparseMatrix_INT32_t_FLOAT64_t(ll_mat, element)
    
        
        elif dtype == FLOAT128_T:
        
            return MakeArrowHeadLLSparseMatrix_INT32_t_FLOAT128_t(ll_mat, element)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakeArrowHeadLLSparseMatrix_INT32_t_COMPLEX64_t(ll_mat, element)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeArrowHeadLLSparseMatrix_INT32_t_COMPLEX128_t(ll_mat, element)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakeArrowHeadLLSparseMatrix_INT32_t_COMPLEX256_t(ll_mat, element)
    
    

    
    elif itype == INT64_T:
    
        
        if dtype == INT32_T:
        
            return MakeArrowHeadLLSparseMatrix_INT64_t_INT32_t(ll_mat, element)
    
        
        elif dtype == INT64_T:
        
            return MakeArrowHeadLLSparseMatrix_INT64_t_INT64_t(ll_mat, element)
    
        
        elif dtype == FLOAT32_T:
        
            return MakeArrowHeadLLSparseMatrix_INT64_t_FLOAT32_t(ll_mat, element)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeArrowHeadLLSparseMatrix_INT64_t_FLOAT64_t(ll_mat, element)
    
        
        elif dtype == FLOAT128_T:
        
            return MakeArrowHeadLLSparseMatrix_INT64_t_FLOAT128_t(ll_mat, element)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakeArrowHeadLLSparseMatrix_INT64_t_COMPLEX64_t(ll_mat, element)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeArrowHeadLLSparseMatrix_INT64_t_COMPLEX128_t(ll_mat, element)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakeArrowHeadLLSparseMatrix_INT64_t_COMPLEX256_t(ll_mat, element)
    
    

    else:
        raise TypeError('itype not recognized')


def DiagonalLLSparseMatrix(**kwargs):
    """
    See ``MakeDiagonalLLSparseMatrix``.

    Note:
        Input arguments are **not** tested.
    """
    element = kwargs.pop('element', None)

    if kwargs.get('store_symmetric', False):
        raise NotImplementedError('This type of matrix is not implemented for symmetric matrices')

    ll_mat = LLSparseMatrix(**kwargs)

    itype = ll_mat.itype
    dtype = ll_mat.dtype

    # create 1.0 element if needed
    if element is None:
        if is_integer_type(dtype):
            element = 1
        elif is_real_type(dtype):
            element = 1.0
        elif is_complex_type(dtype):
            element = 1.0 + 0.0j
        else:
            raise TypeError('dtype not recognized')

    # launch right "constructor" method

    
    if itype == INT32_T:
    
        
        if dtype == INT32_T:
        
            return MakeDiagonalLLSparseMatrix_INT32_t_INT32_t(ll_mat, element)
    
        
        elif dtype == INT64_T:
        
            return MakeDiagonalLLSparseMatrix_INT32_t_INT64_t(ll_mat, element)
    
        
        elif dtype == FLOAT32_T:
        
            return MakeDiagonalLLSparseMatrix_INT32_t_FLOAT32_t(ll_mat, element)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeDiagonalLLSparseMatrix_INT32_t_FLOAT64_t(ll_mat, element)
    
        
        elif dtype == FLOAT128_T:
        
            return MakeDiagonalLLSparseMatrix_INT32_t_FLOAT128_t(ll_mat, element)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakeDiagonalLLSparseMatrix_INT32_t_COMPLEX64_t(ll_mat, element)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeDiagonalLLSparseMatrix_INT32_t_COMPLEX128_t(ll_mat, element)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakeDiagonalLLSparseMatrix_INT32_t_COMPLEX256_t(ll_mat, element)
    
    

    
    elif itype == INT64_T:
    
        
        if dtype == INT32_T:
        
            return MakeDiagonalLLSparseMatrix_INT64_t_INT32_t(ll_mat, element)
    
        
        elif dtype == INT64_T:
        
            return MakeDiagonalLLSparseMatrix_INT64_t_INT64_t(ll_mat, element)
    
        
        elif dtype == FLOAT32_T:
        
            return MakeDiagonalLLSparseMatrix_INT64_t_FLOAT32_t(ll_mat, element)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeDiagonalLLSparseMatrix_INT64_t_FLOAT64_t(ll_mat, element)
    
        
        elif dtype == FLOAT128_T:
        
            return MakeDiagonalLLSparseMatrix_INT64_t_FLOAT128_t(ll_mat, element)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakeDiagonalLLSparseMatrix_INT64_t_COMPLEX64_t(ll_mat, element)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeDiagonalLLSparseMatrix_INT64_t_COMPLEX128_t(ll_mat, element)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakeDiagonalLLSparseMatrix_INT64_t_COMPLEX256_t(ll_mat, element)
    
    

    else:
        raise TypeError('itype not recognized')

# alias
def IdentityLLSparseMatrix(**kwargs):
    element = kwargs.pop('element', None)
    return DiagonalLLSparseMatrix(**kwargs)


def BandLLSparseMatrix(**kwargs):
    """
    See ``MakeBandLLSparseMatrix_INT32_t_INT32_t``.

    Note:
        Input arguments are **not** tested.
    """
    diag_coeff = kwargs.pop('diag_coeff', None)
    numpy_arrays = kwargs.pop('numpy_arrays', None)

    if diag_coeff is None:
        raise ValueError("Named argument 'diag_coeff' not given")

    if numpy_arrays is None:
        raise ValueError("Named argument 'numpy_arrays' not given")

    ll_mat = LLSparseMatrix(**kwargs)

    itype = ll_mat.itype
    dtype = ll_mat.dtype

    # launch right "constructor" method

    
    if itype == INT32_T:
    
        
        if dtype == INT32_T:
        
            return MakeBandLLSparseMatrix_INT32_t_INT32_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == INT64_T:
        
            return MakeBandLLSparseMatrix_INT32_t_INT64_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == FLOAT32_T:
        
            return MakeBandLLSparseMatrix_INT32_t_FLOAT32_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeBandLLSparseMatrix_INT32_t_FLOAT64_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == FLOAT128_T:
        
            return MakeBandLLSparseMatrix_INT32_t_FLOAT128_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakeBandLLSparseMatrix_INT32_t_COMPLEX64_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeBandLLSparseMatrix_INT32_t_COMPLEX128_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakeBandLLSparseMatrix_INT32_t_COMPLEX256_t(ll_mat, diag_coeff, numpy_arrays)
    
    

    
    elif itype == INT64_T:
    
        
        if dtype == INT32_T:
        
            return MakeBandLLSparseMatrix_INT64_t_INT32_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == INT64_T:
        
            return MakeBandLLSparseMatrix_INT64_t_INT64_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == FLOAT32_T:
        
            return MakeBandLLSparseMatrix_INT64_t_FLOAT32_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeBandLLSparseMatrix_INT64_t_FLOAT64_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == FLOAT128_T:
        
            return MakeBandLLSparseMatrix_INT64_t_FLOAT128_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakeBandLLSparseMatrix_INT64_t_COMPLEX64_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeBandLLSparseMatrix_INT64_t_COMPLEX128_t(ll_mat, diag_coeff, numpy_arrays)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakeBandLLSparseMatrix_INT64_t_COMPLEX256_t(ll_mat, diag_coeff, numpy_arrays)
    
    

    else:
        raise TypeError('itype not recognized')

def LinearFillLLSparseMatrix(**kwargs):
    """
    See ``MakeLinearFillLLSparseMatrix``.

    Note:
        Input arguments are **not** tested.
    """
    first_element = kwargs.pop('first_element', None)
    step = kwargs.pop('step', None)
    row_wise = kwargs.pop('row_wise', True)

    # this is only really needed to compute 'size_hint'
    size = kwargs.get('size', None)
    if size is None:
        nrow = kwargs.get('nrow', None)
        if nrow is None:
            raise TypeError('We need a size to construct the matrix')
        ncol = kwargs.get('ncol', None)
        if ncol is None:
            raise TypeError('We need a size to construct the matrix')
    else:
        nrow = ncol = size

    kwargs['size_hint'] = nrow * ncol

    ll_mat = LLSparseMatrix(**kwargs)

    itype = ll_mat.itype
    dtype = ll_mat.dtype

    # create 1.0 first element and step if needed
    if first_element is None or step is None:
        if is_integer_type(dtype):
            if first_element is None:
                first_element = 1
            if step is None:
                step = 1
        elif is_real_type(dtype):
            if first_element is None:
                first_element = 1.0
            if step is None:
                step = 1.0
        elif is_complex_type(dtype):
            if first_element is None:
                first_element = 1.0 + 0.0j
            if step is None:
                step = 1.0 + 0.0j
        else:
            raise TypeError('dtype not recognized')

    # launch right "constructor" method

    
    if itype == INT32_T:
    
        
        if dtype == INT32_T:
        
            return MakeLinearFillLLSparseMatrix_INT32_t_INT32_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == INT64_T:
        
            return MakeLinearFillLLSparseMatrix_INT32_t_INT64_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == FLOAT32_T:
        
            return MakeLinearFillLLSparseMatrix_INT32_t_FLOAT32_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeLinearFillLLSparseMatrix_INT32_t_FLOAT64_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == FLOAT128_T:
        
            return MakeLinearFillLLSparseMatrix_INT32_t_FLOAT128_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakeLinearFillLLSparseMatrix_INT32_t_COMPLEX64_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeLinearFillLLSparseMatrix_INT32_t_COMPLEX128_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakeLinearFillLLSparseMatrix_INT32_t_COMPLEX256_t(ll_mat, first_element, step, row_wise)
    
    

    
    elif itype == INT64_T:
    
        
        if dtype == INT32_T:
        
            return MakeLinearFillLLSparseMatrix_INT64_t_INT32_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == INT64_T:
        
            return MakeLinearFillLLSparseMatrix_INT64_t_INT64_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == FLOAT32_T:
        
            return MakeLinearFillLLSparseMatrix_INT64_t_FLOAT32_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == FLOAT64_T:
        
            return MakeLinearFillLLSparseMatrix_INT64_t_FLOAT64_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == FLOAT128_T:
        
            return MakeLinearFillLLSparseMatrix_INT64_t_FLOAT128_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakeLinearFillLLSparseMatrix_INT64_t_COMPLEX64_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakeLinearFillLLSparseMatrix_INT64_t_COMPLEX128_t(ll_mat, first_element, step, row_wise)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakeLinearFillLLSparseMatrix_INT64_t_COMPLEX256_t(ll_mat, first_element, step, row_wise)
    
    

    else:
        raise TypeError('itype not recognized')

def PermutationLLSparseMatrix(**kwargs):
    """
    See ``MakePermutationLLSparseMatrix``.

    Note:
        Input arguments are **not** tested.
    """
    p_vec = kwargs.pop('P', None)

    if kwargs.get('store_symmetric', False):
        raise NotImplementedError('This type of matrix is not implemented for symmetric matrices')

    ll_mat = LLSparseMatrix(**kwargs)

    itype = ll_mat.itype
    dtype = ll_mat.dtype

    # launch right "constructor" method

    
    if itype == INT32_T:
    
        
        if dtype == INT32_T:
        
            return MakePermutationLLSparseMatrix_INT32_t_INT32_t(ll_mat, p_vec)
    
        
        elif dtype == INT64_T:
        
            return MakePermutationLLSparseMatrix_INT32_t_INT64_t(ll_mat, p_vec)
    
        
        elif dtype == FLOAT32_T:
        
            return MakePermutationLLSparseMatrix_INT32_t_FLOAT32_t(ll_mat, p_vec)
    
        
        elif dtype == FLOAT64_T:
        
            return MakePermutationLLSparseMatrix_INT32_t_FLOAT64_t(ll_mat, p_vec)
    
        
        elif dtype == FLOAT128_T:
        
            return MakePermutationLLSparseMatrix_INT32_t_FLOAT128_t(ll_mat, p_vec)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakePermutationLLSparseMatrix_INT32_t_COMPLEX64_t(ll_mat, p_vec)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakePermutationLLSparseMatrix_INT32_t_COMPLEX128_t(ll_mat, p_vec)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakePermutationLLSparseMatrix_INT32_t_COMPLEX256_t(ll_mat, p_vec)
    
    

    
    elif itype == INT64_T:
    
        
        if dtype == INT32_T:
        
            return MakePermutationLLSparseMatrix_INT64_t_INT32_t(ll_mat, p_vec)
    
        
        elif dtype == INT64_T:
        
            return MakePermutationLLSparseMatrix_INT64_t_INT64_t(ll_mat, p_vec)
    
        
        elif dtype == FLOAT32_T:
        
            return MakePermutationLLSparseMatrix_INT64_t_FLOAT32_t(ll_mat, p_vec)
    
        
        elif dtype == FLOAT64_T:
        
            return MakePermutationLLSparseMatrix_INT64_t_FLOAT64_t(ll_mat, p_vec)
    
        
        elif dtype == FLOAT128_T:
        
            return MakePermutationLLSparseMatrix_INT64_t_FLOAT128_t(ll_mat, p_vec)
    
        
        elif dtype == COMPLEX64_T:
        
            return MakePermutationLLSparseMatrix_INT64_t_COMPLEX64_t(ll_mat, p_vec)
    
        
        elif dtype == COMPLEX128_T:
        
            return MakePermutationLLSparseMatrix_INT64_t_COMPLEX128_t(ll_mat, p_vec)
    
        
        elif dtype == COMPLEX256_T:
        
            return MakePermutationLLSparseMatrix_INT64_t_COMPLEX256_t(ll_mat, p_vec)
    
    

    else:
        raise TypeError('itype not recognized')