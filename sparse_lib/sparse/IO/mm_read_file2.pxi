from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

from libc.stdio cimport *
from libc.string cimport *
from cpython.unicode cimport *

cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile2(str filename, bint store_zeros=False, bint test_bounds=False):
    """

    """
    # TODO: this version is painly slow...
    cdef:
        str line
        list token_list
        str token
        char data_type
        char storage_scheme
        SIZE_t nrow
        SIZE_t ncol
        SIZE_t nnz
        SIZE_t nnz_read

    cdef:
        bint sparse
        bint is_symmetric
        bint is_complex
        list sparse_dense_list = [MM_ARRAY_STR, MM_COORDINATE_STR]
        list data_type_list = [MM_COMPLEX_STR, MM_REAL_STR, MM_INT_STR, MM_PATTERN_STR]
        dict data_type_dict = {MM_COMPLEX_STR : COMPLEX, MM_REAL_STR : REAL, MM_INT_STR : INTEGER, MM_PATTERN_STR : PATTERN}
        list storage_scheme_list = [MM_GENERAL_STR, MM_SYMM_STR, MM_HERM_STR, MM_SKEW_STR]
        dict storage_scheme_dict = {MM_GENERAL_STR : GENERAL, MM_SYMM_STR : SYMMETRIC, MM_HERM_STR : HERMITIAN, MM_SKEW_STR : SKEW}

    cdef LLSparseMatrix A

    with open(filename, 'r') as f:
        # read banner
        line = f.readline()
        token_list = line.split()

        if len(token_list) != 5:
            raise IOError('Matrix format not recognized as Matrix Market format: not the right number of tokens in the Matrix Market banner')

        # MATRIX MARKET BANNER START
        if token_list[0] != MM_MATRIX_MARKET_BANNER_STR:
            raise IOError('Matrix format not recognized as Matrix Market format: zeroth token in the Matrix Market banner is not "%s"' % MM_MATRIX_MARKET_BANNER_STR)

        # OBJECT
        if token_list[1].lower() != MM_MTX_STR:
            raise IOError('Matrix format not recognized as Matrix Market format: first token in the Matrix Market banner is not "%s"' % MM_MTX_STR)

        # SPARSE/DENSE
        token = token_list[2].lower()
        if token not in sparse_dense_list:
            raise IOError('Matrix format not recognized as Matrix Market format: third token in the Matrix Market banner is not in "%s"' % sparse_dense_list)

        if token == MM_ARRAY_STR:
            sparse = False
        else:
            sparse = True

        # DATA TYPE
        token = token_list[3].lower()
        if token not in data_type_list:
            raise IOError('Matrix format not recognized as Matrix Market format: fourth token in the Matrix Market banner is not in "%s"' % data_type_list)
        data_type = data_type_dict[token]
        is_complex = data_type == COMPLEX


        # STORAGE SCHEME
        token = token_list[4].lower()
        if token not in storage_scheme_list:
            raise IOError('Matrix format not recognized as Matrix Market format: fourth token in the Matrix Market banner is not in "%s"' % storage_scheme_list)
        storage_scheme = storage_scheme_dict[token]
        is_symmetric = storage_scheme == SYMMETRIC

        # SKIP COMMENTS
        line = f.readline()
        while line[0] == '%':
            line = f.readline()

        # READ DATA SPECIFICATION
        token_list = line.split()
        if len(token_list) != 3:
            raise IOError('Matrix format not recognized as Matrix Market format: line right after comments must contain (nrow, ncol, nnz)')

        nrow = atoi(token_list[0])
        ncol = atoi(token_list[1])
        nnz = atoi(token_list[2])

        if data_type == PATTERN:
            raise IOError('Matrix Market format not supported for PATTERN')

        A = LLSparseMatrix(nrow=nrow, ncol=ncol, size_hint=nnz, is_symmetric=is_symmetric, is_complex=is_complex, store_zeros=store_zeros)


        line = f.readline()
        nnz_read = 0
        if test_bounds:
            while line:
                nnz_read += 1
                token_list = line.split()

                if is_complex:
                    A.safe_put(atoi(token_list[0])-1, atoi(token_list[1]) - 1, atof(token_list[2]), atof(token_list[3]))
                else:
                    A.safe_put(atoi(token_list[0])-1, atoi(token_list[1]) - 1, atof(token_list[2]))

                line = f.readline()

        else:
            while line:
                nnz_read += 1
                token_list = line.split()

                if is_complex:
                    A.put(atoi(token_list[0])-1, atoi(token_list[1]) - 1, atof(token_list[2]), atof(token_list[3]))
                else:
                    A.put(atoi(token_list[0])-1, atoi(token_list[1]) - 1, atof(token_list[2]))

                line = f.readline()

        if nnz != nnz_read:
            raise IOError('Matrix Market file contains %d data lines instead of %d' % (nnz_read, nnz))

    return A



########################################################################################################################
# Matrix Market I/O library
########################################################################################################################











