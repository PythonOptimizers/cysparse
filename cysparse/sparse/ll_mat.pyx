
#from cysparse.types.cysparse_types cimport *
from cysparse.sparse.ll_mat cimport LL_MAT_DEFAULT_SIZE_HINT
from cysparse.sparse.sparse_mat cimport unexposed_value

from cysparse.types.cysparse_types import BASIC_TYPES

LL_MAT_DEFAULT_SIZE_HINT = 40


    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT32_t cimport LLSparseMatrix_INT32_t_INT32_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_INT64_t cimport LLSparseMatrix_INT32_t_INT64_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT32_t cimport LLSparseMatrix_INT32_t_FLOAT32_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT32_t_FLOAT64_t cimport LLSparseMatrix_INT32_t_FLOAT64_t
    

    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_INT32_t cimport LLSparseMatrix_INT64_t_INT32_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_INT64_t cimport LLSparseMatrix_INT64_t_INT64_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT32_t cimport LLSparseMatrix_INT64_t_FLOAT32_t
    
from cysparse.sparse.ll_mat_matrices.ll_mat_INT64_t_FLOAT64_t cimport LLSparseMatrix_INT64_t_FLOAT64_t
    


def NewLLSparseMatrix(**kwargs):
    """
    Factory method to create an LLSparseMatrix.
    """
    ####################################################################################################################
    #                                            *** Get arguments ***
    ####################################################################################################################
    # General specifications
    nrow = kwargs.get('nrow', -1)
    ncol = kwargs.get('ncol', -1)
    size = kwargs.get('size', -1)
    size_hint = kwargs.get('size_hint', LL_MAT_DEFAULT_SIZE_HINT)

    itype_str = kwargs.get('itype', 'INT32_t')
    dtype_str = kwargs.get('dtype', 'FLOAT64_t')

    assert itype_str in BASIC_TYPES, "itype string '%s' not recognized" % itype_str
    assert dtype_str in BASIC_TYPES, "dtype string '%s' not recognized" % dtype_str

    itype = BASIC_TYPES[itype_str][1]
    dtype = BASIC_TYPES[dtype_str][1]

    assert itype in [INT32_T,INT64_T], "itype is not accepted as index type"
    assert dtype in [INT32_T,INT64_T,FLOAT32_T,FLOAT64_T], "dtype is not accepted as type for a matrix element"

    cdef bint store_zeros = kwargs.get('store_zeros', False)
    cdef bint is_symmetric = kwargs.get('is_symmetric', False)
    cdef bint is_complex = kwargs.get('is_complex', False)
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
        
                return LLSparseMatrix_INT32_t_INT32_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)
    
        
            elif dtype == INT64_T:
        
                return LLSparseMatrix_INT32_t_INT64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)
    
        
            elif dtype == FLOAT32_T:
        
                return LLSparseMatrix_INT32_t_FLOAT32_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)
    
        
            elif dtype == FLOAT64_T:
        
                return LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)
    

    
        elif itype == INT64_T:
    
    
        
            if dtype == INT32_T:
        
                return LLSparseMatrix_INT64_t_INT32_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)
    
        
            elif dtype == INT64_T:
        
                return LLSparseMatrix_INT64_t_INT64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)
    
        
            elif dtype == FLOAT32_T:
        
                return LLSparseMatrix_INT64_t_FLOAT32_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)
    
        
            elif dtype == FLOAT64_T:
        
                return LLSparseMatrix_INT64_t_FLOAT64_t(control_object=unexposed_value, nrow=real_nrow, ncol=real_ncol, size_hint=size_hint, store_zeros=store_zeros, is_symmetric=is_symmetric, is_complex=is_complex)
    


    #                                            CASE 1: from another matrix
    if matrix is not None:
        raise NotImplementedError("Cannot create a LLSparseMatrix from another matrix (yet)")

    if from_filename:
        raise NotImplementedError("Cannot create a LLSparseMatrix from a file (yet)")