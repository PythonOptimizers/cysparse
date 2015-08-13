"""
Matrix Market IO

See http://math.nist.gov/MatrixMarket/ .

"""


cdef LLSparseMatrix_INT32_t_FLOAT64_t MakeLLSparseMatrixFromMMFile_INT32_t_FLOAT64_t(str mm_filename, bint store_zeros=False, bint test_bounds=True):
    cdef:
        str line
        list token_list
        str token
        char data_type
        char storage_scheme
        INT32_t nrow
        INT32_t ncol
        INT32_t nnz
        INT32_t nnz_read

    cdef:
        bint sparse
        bint is_symmetric
        bint is_complex
        list sparse_dense_list = [MM_ARRAY_STR, MM_COORDINATE_STR]
        list data_type_list = [MM_COMPLEX_STR, MM_REAL_STR, MM_INT_STR, MM_PATTERN_STR]
        dict data_type_dict = {MM_COMPLEX_STR : MM_COMPLEX, MM_REAL_STR : MM_REAL, MM_INT_STR : MM_INTEGER, MM_PATTERN_STR : MM_PATTERN}
        list storage_scheme_list = [MM_GENERAL_STR, MM_SYMM_STR, MM_HERM_STR, MM_SKEW_STR]
        dict storage_scheme_dict = {MM_GENERAL_STR : MM_GENERAL, MM_SYMM_STR : MM_SYMMETRIC, MM_HERM_STR : MM_HERMITIAN, MM_SKEW_STR : MM_SKEW}

        COMPLEX128_t z, w
        FLOAT64_t real_part, imag_part

    cdef LLSparseMatrix_INT32_t_FLOAT64_t A

    with open(mm_filename, 'r') as f:

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

        # TODO: test if matrix is sparse or not
        if token == MM_ARRAY_STR:
            sparse = False
        else:
            sparse = True

        if not sparse:
            raise IOError('CySparse only read sparse matrices')

        # DATA TYPE
        token = token_list[3].lower()
        if token not in data_type_list:
            raise IOError('Matrix format not recognized as Matrix Market format: fourth token in the Matrix Market banner is not in "%s"' % data_type_list)
        data_type = data_type_dict[token]
        is_complex = data_type == MM_COMPLEX


        # STORAGE SCHEME
        token = token_list[4].lower()
        if token not in storage_scheme_list:
            raise IOError('Matrix format not recognized as Matrix Market format: fourth token in the Matrix Market banner is not in "%s"' % storage_scheme_list)
        storage_scheme = storage_scheme_dict[token]
        is_symmetric = storage_scheme == MM_SYMMETRIC

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

        if data_type == MM_PATTERN:
            raise IOError('Matrix Market format not supported for PATTERN')

        A = LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value, nrow=nrow, ncol=ncol, size_hint=nnz, is_symmetric=is_symmetric, is_complex=is_complex, store_zeros=store_zeros)


        line = f.readline()
        nnz_read = 0
        if test_bounds:
            while line:
                nnz_read += 1
                token_list = line.split()

                A.safe_put(atoi(token_list[0])-1, atoi(token_list[1]) - 1, <FLOAT64_t>atof(token_list[2]))

                line = f.readline()

        else: # don't test bounds
            while line:
                nnz_read += 1
                token_list = line.split()

                A.put(atoi(token_list[0])-1, atoi(token_list[1]) - 1, <FLOAT64_t>atof(token_list[2]))

                line = f.readline()

        if nnz != nnz_read:
            raise IOError('Matrix Market file contains %d data lines instead of %d' % (nnz_read, nnz))

    return A

########################################################################################################################
# Optimized version
########################################################################################################################
# This version doesn't work yet...
# TODO: write this!
cdef LLSparseMatrix_INT32_t_FLOAT64_t MakeLLSparseMatrixFromMMFile2_INT32_t_FLOAT64_t(str mm_filename, bint store_zeros=False, bint test_bounds=True):
    cdef:
        str line
        list token_list
        str token
        char data_type
        char storage_scheme
        INT32_t nrow
        INT32_t ncol
        INT32_t nnz
        INT32_t nnz_read

    cdef:
        bint sparse
        bint is_symmetric
        bint is_complex
        list sparse_dense_list = [MM_ARRAY_STR, MM_COORDINATE_STR]
        list data_type_list = [MM_COMPLEX_STR, MM_REAL_STR, MM_INT_STR, MM_PATTERN_STR]
        dict data_type_dict = {MM_COMPLEX_STR : MM_COMPLEX, MM_REAL_STR : MM_REAL, MM_INT_STR : MM_INTEGER, MM_PATTERN_STR : MM_PATTERN}
        list storage_scheme_list = [MM_GENERAL_STR, MM_SYMM_STR, MM_HERM_STR, MM_SKEW_STR]
        dict storage_scheme_dict = {MM_GENERAL_STR : MM_GENERAL, MM_SYMM_STR : MM_SYMMETRIC, MM_HERM_STR : MM_HERMITIAN, MM_SKEW_STR : MM_SKEW}



    cdef LLSparseMatrix_INT32_t_FLOAT64_t A

    cdef:
        INT32_t free, i, j
        FLOAT64_t v

        FLOAT64_t  *val
        INT32_t *col
        INT32_t *link
        INT32_t *root
        INT32_t *end_root

    with open(mm_filename, 'r') as f:

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

        # TODO: test if matrix is sparse or not
        if token == MM_ARRAY_STR:
            sparse = False
        else:
            sparse = True

        if not sparse:
            raise IOError('CySparse only read sparse matrices')

        # DATA TYPE
        token = token_list[3].lower()
        if token not in data_type_list:
            raise IOError('Matrix format not recognized as Matrix Market format: fourth token in the Matrix Market banner is not in "%s"' % data_type_list)
        data_type = data_type_dict[token]
        is_complex = data_type == MM_COMPLEX


        # STORAGE SCHEME
        token = token_list[4].lower()
        if token not in storage_scheme_list:
            raise IOError('Matrix format not recognized as Matrix Market format: fourth token in the Matrix Market banner is not in "%s"' % storage_scheme_list)
        storage_scheme = storage_scheme_dict[token]
        is_symmetric = storage_scheme == MM_SYMMETRIC

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

        if data_type == MM_PATTERN:
            raise IOError('Matrix Market format not supported for PATTERN')

        A = LLSparseMatrix_INT32_t_FLOAT64_t(control_object=unexposed_value,
                                          no_memory=True,
                                          nrow=nrow,
                                          ncol=ncol,
                                          size_hint=nnz,
                                          is_symmetric=is_symmetric,
                                          store_zeros=store_zeros)



        val = <FLOAT64_t *> PyMem_Malloc(nnz * sizeof(FLOAT64_t))
        if not val:
            raise MemoryError()

        col = <INT32_t *> PyMem_Malloc(nnz * sizeof(INT32_t))
        if not col:
            PyMem_Free(val)
            raise MemoryError()

        link = <INT32_t *> PyMem_Malloc(nnz * sizeof(INT32_t))
        if not link:
            PyMem_Free(val)
            PyMem_Free(col)
            raise MemoryError()

        root = <INT32_t *> PyMem_Malloc(nrow * sizeof(INT32_t))
        if not root:
            PyMem_Free(val)
            PyMem_Free(col)
            PyMem_Free(link)
            raise MemoryError()

        end_root = <INT32_t *> PyMem_Malloc(nrow * sizeof(INT32_t))
        if not end_root:
            PyMem_Free(val)
            PyMem_Free(col)
            PyMem_Free(link)
            PyMem_Free(root)
            raise MemoryError()

        for i from 0 <= i < nrow:
            root[i] = -1
            end_root[i] = -1

        ### DOES NOT WORK (YET) ###
        # Reading matrix content
        for nnz_read from 0 <= nnz_read < nnz:
            print "read element... "
            #############################
            # read new element (i, j, v)
            #############################
            line = f.readline()
            token_list = line.split()
            i = atoi(token_list[0])-1
            j = atoi(token_list[1]) - 1

            #print "i = %d" % i
            #print "j = %d" % j

            if test_bounds:
                if not (0 <= i < nrow):
                    raise IndexTypeError('Value of index i=%d is out of bounds (%d, %d)' % (i, 0, nrow))
                if not (0 <= j < ncol):
                    raise IndexTypeError('Value of index j=%d is out of bounds (%d, %d)' % (j, 0, ncol))


            v = <FLOAT64_t>atof(token_list[2])


            #############################
            # fill in arrays
            #############################

            if root[i] == -1:
                print "first element on row i"
                # first element on row i
                root[i] = nnz_read

            col[nnz_read] = j
            val[nnz_read] = v
            # last element on row i
            end_root[i] = nnz_read
            link[end_root[i]] = nnz_read

        # post processing
        # close row lists
        for i from 0 <= i < nrow:
            if end_root[i] != -1:
                link[end_root[i]] = -1

        # updating the matrix
        A.col = col
        A.val = val
        A.link = link
        A.root = root
        A.free = -1
        A.__nnz = nnz
        A.nalloc = nnz


        if nnz != nnz_read:
            raise IOError('Matrix Market file contains %d data lines instead of %d' % (nnz_read, nnz))

    return A
