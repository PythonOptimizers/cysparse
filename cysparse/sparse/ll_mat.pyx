
from cysparse.sparse.ll_mat cimport LL_MAT_DEFAULT_SIZE_HINT
from cysparse.sparse.s_mat cimport unexposed_value

from cysparse.types.cysparse_types import *

from cysparse.sparse.s_mat cimport SparseMatrix

from cython cimport isinstance
from libc.stdio cimport *
from libc.string cimport *
from cpython.unicode cimport *
from libc.stdlib cimport * 


LL_MAT_INCREASE_FACTOR = 1.5
LL_MAT_DEFAULT_SIZE_HINT = 40

LL_MAT_PPRINT_COL_THRESH = 20
LL_MAT_PPRINT_ROW_THRESH = 40


    
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
# Factory methods
########################################################################################################################
def NewLLSparseMatrix(**kwargs):
    """
    Factory method to create an ``LLSparseMatrix``.
    """
    ####################################################################################################################
    #                                            *** Get arguments ***
    ####################################################################################################################
    # General specifications
    nrow = kwargs.get('nrow', -1)
    ncol = kwargs.get('ncol', -1)
    size = kwargs.get('size', -1)
    size_hint = kwargs.get('size_hint', LL_MAT_DEFAULT_SIZE_HINT)

    itype = kwargs.get('itype', INT32_T)
    dtype = kwargs.get('dtype', FLOAT64_T)

    assert itype in INDEX_TYPES, "itype not recognized"
    assert dtype in ELEMENT_TYPES, "dtype not recognized"

    assert itype in [INT32_T,INT64_T], "itype is not accepted as index type"
    assert dtype in [INT32_T,INT64_T,FLOAT32_T,FLOAT64_T,FLOAT128_T,COMPLEX64_T,COMPLEX128_T,COMPLEX256_T], "dtype is not accepted as type for a matrix element"

    cdef bint store_zeros = kwargs.get('store_zeros', False)
    cdef bint is_symmetric = kwargs.get('is_symmetric', False)
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
        
                return LLSparseMatrix_INT32_t_INT32_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == INT64_T:
        
                return LLSparseMatrix_INT32_t_INT64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == FLOAT32_T:
        
                return LLSparseMatrix_INT32_t_FLOAT32_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == FLOAT64_T:
        
                return LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == FLOAT128_T:
        
                return LLSparseMatrix_INT32_t_FLOAT128_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == COMPLEX64_T:
        
                return LLSparseMatrix_INT32_t_COMPLEX64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == COMPLEX128_T:
        
                return LLSparseMatrix_INT32_t_COMPLEX128_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == COMPLEX256_T:
        
                return LLSparseMatrix_INT32_t_COMPLEX256_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    

    
        elif itype == INT64_T:
    
    
        
            if dtype == INT32_T:
        
                return LLSparseMatrix_INT64_t_INT32_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == INT64_T:
        
                return LLSparseMatrix_INT64_t_INT64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == FLOAT32_T:
        
                return LLSparseMatrix_INT64_t_FLOAT32_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == FLOAT64_T:
        
                return LLSparseMatrix_INT64_t_FLOAT64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == FLOAT128_T:
        
                return LLSparseMatrix_INT64_t_FLOAT128_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == COMPLEX64_T:
        
                return LLSparseMatrix_INT64_t_COMPLEX64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == COMPLEX128_T:
        
                return LLSparseMatrix_INT64_t_COMPLEX128_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    
        
            elif dtype == COMPLEX256_T:
        
                return LLSparseMatrix_INT64_t_COMPLEX256_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric)
    


    #                                            CASE 2: from another matrix
    if matrix is not None:
        raise NotImplementedError("Cannot create a LLSparseMatrix from another matrix (yet)")

    #                                            CASE 3: from a file
    if from_filename:
        if mm_filename is not None:
            # Matrix Market format

            assert itype in [INT32_T,INT64_T], "itype is not accepted as index type for a matrix from a Matrix Market file"

            assert dtype in [INT64_T,FLOAT64_T,COMPLEX128_T], "dtype is not accepted as type for a matrix from a Matrix Market file"


    
            if itype == INT32_T:
    
    
        
                if dtype == INT64_T:
        
                    return MakeLLSparseMatrixFromMMFile_INT32_t_INT64_t(mm_filename=mm_filename, store_zeros=store_zeros, test_bounds=test_bounds)
    
        
                elif dtype == FLOAT64_T:
        
                    return MakeLLSparseMatrixFromMMFile_INT32_t_FLOAT64_t(mm_filename=mm_filename, store_zeros=store_zeros, test_bounds=test_bounds)
    
        
                elif dtype == COMPLEX128_T:
        
                    return MakeLLSparseMatrixFromMMFile_INT32_t_COMPLEX128_t(mm_filename=mm_filename, store_zeros=store_zeros, test_bounds=test_bounds)
    

    
            elif itype == INT64_T:
    
    
        
                if dtype == INT64_T:
        
                    return MakeLLSparseMatrixFromMMFile_INT64_t_INT64_t(mm_filename=mm_filename, store_zeros=store_zeros, test_bounds=test_bounds)
    
        
                elif dtype == FLOAT64_T:
        
                    return MakeLLSparseMatrixFromMMFile_INT64_t_FLOAT64_t(mm_filename=mm_filename, store_zeros=store_zeros, test_bounds=test_bounds)
    
        
                elif dtype == COMPLEX128_T:
        
                    return MakeLLSparseMatrixFromMMFile_INT64_t_COMPLEX128_t(mm_filename=mm_filename, store_zeros=store_zeros, test_bounds=test_bounds)
    





def NewLLSparseMatrixFromMMFile(filename):
    """
    Factory method to create an ``LLSparseMatrix`` from a ``Matrix Market`` file.

    Return the minimalistic ``LLSparseMatrix`` possible.
    """
    pass