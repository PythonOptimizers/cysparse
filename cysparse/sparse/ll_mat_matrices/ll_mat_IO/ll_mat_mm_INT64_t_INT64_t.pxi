"""
Matrix Market IO

See http://math.nist.gov/MatrixMarket/ .

"""


cdef LLSparseMatrix_INT64_t_INT64_t MakeLLSparseMatrixFromMMFile_INT64_t_INT64_t(str mm_filename, bint use_zero_storage=False, bint test_bounds=True):
    cdef:
        str line
        list token_list
        str token
        char data_type
        char storage_scheme
        INT64_t nrow
        INT64_t ncol
        INT64_t nnz
        INT64_t nnz_read

    cdef:
        bint sparse
        bint use_symmetric_storage
        bint is_complex
        list sparse_dense_list = [MM_ARRAY_STR, MM_COORDINATE_STR]
        list data_type_list = [MM_COMPLEX_STR, MM_REAL_STR, MM_INT_STR, MM_PATTERN_STR]
        dict data_type_dict = {MM_COMPLEX_STR : MM_COMPLEX, MM_REAL_STR : MM_REAL, MM_INT_STR : MM_INTEGER, MM_PATTERN_STR : MM_PATTERN}
        list storage_scheme_list = [MM_GENERAL_STR, MM_SYMM_STR, MM_HERM_STR, MM_SKEW_STR]
        dict storage_scheme_dict = {MM_GENERAL_STR : MM_GENERAL, MM_SYMM_STR : MM_SYMMETRIC, MM_HERM_STR : MM_HERMITIAN, MM_SKEW_STR : MM_SKEW}

        COMPLEX128_t z, w
        FLOAT64_t real_part, imag_part

    cdef LLSparseMatrix_INT64_t_INT64_t A

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
        use_symmetric_storage = storage_scheme == MM_SYMMETRIC

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

        A = LLSparseMatrix_INT64_t_INT64_t(control_object=unexposed_value,
                                          nrow=nrow,
                                          ncol=ncol,
                                          size_hint=nnz,
                                          use_symmetric_storage=use_symmetric_storage,
                                          use_zero_storage=use_zero_storage)

        line = f.readline()
        nnz_read = 0
        if test_bounds:
            while line:
                nnz_read += 1
                token_list = line.split()

                A.safe_put(atoi(token_list[0])-1, atoi(token_list[1]) - 1, <INT64_t>atoi(token_list[2]))

                line = f.readline()

        else: # don't test bounds
            while line:
                nnz_read += 1
                token_list = line.split()

                A.put(atoi(token_list[0])-1, atoi(token_list[1]) - 1, <INT64_t>atoi(token_list[2]))

                line = f.readline()

        if nnz != nnz_read:
            raise IOError('Matrix Market file contains %d data lines instead of %d' % (nnz_read, nnz))

    return A

########################################################################################################################
# Optimized version
########################################################################################################################
# This version doesn't work yet...
# TODO: write this!
cdef LLSparseMatrix_INT64_t_INT64_t MakeLLSparseMatrixFromMMFile2_INT64_t_INT64_t(str mm_filename, bint use_zero_storage=False, bint test_bounds=True):
    cdef:
        str line
        list token_list
        str token
        char data_type
        char storage_scheme
        INT64_t nrow
        INT64_t ncol
        INT64_t nnz
        INT64_t nnz_real
        INT64_t nnz_read


    cdef:
        bint sparse
        bint use_symmetric_storage
        bint is_complex
        list sparse_dense_list = [MM_ARRAY_STR, MM_COORDINATE_STR]
        list data_type_list = [MM_COMPLEX_STR, MM_REAL_STR, MM_INT_STR, MM_PATTERN_STR]
        dict data_type_dict = {MM_COMPLEX_STR : MM_COMPLEX, MM_REAL_STR : MM_REAL, MM_INT_STR : MM_INTEGER, MM_PATTERN_STR : MM_PATTERN}
        list storage_scheme_list = [MM_GENERAL_STR, MM_SYMM_STR, MM_HERM_STR, MM_SKEW_STR]
        dict storage_scheme_dict = {MM_GENERAL_STR : MM_GENERAL, MM_SYMM_STR : MM_SYMMETRIC, MM_HERM_STR : MM_HERMITIAN, MM_SKEW_STR : MM_SKEW}

        COMPLEX128_t z, w
        FLOAT64_t real_part, imag_part

    cdef LLSparseMatrix_INT64_t_INT64_t A

    cdef:
        INT64_t free, i, j
        INT64_t v

        INT64_t  *val
        INT64_t *col
        INT64_t *link
        INT64_t *root
        INT64_t *end_root

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
        use_symmetric_storage = storage_scheme == MM_SYMMETRIC

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

        A = LLSparseMatrix_INT64_t_INT64_t(control_object=unexposed_value,
                                          no_memory=True,
                                          nrow=nrow,
                                          ncol=ncol,
                                          size_hint=nnz,
                                          use_symmetric_storage=use_symmetric_storage,
                                          use_zero_storage=use_zero_storage)



        val = <INT64_t *> PyMem_Malloc(nnz * sizeof(INT64_t))
        if not val:
            raise MemoryError()

        col = <INT64_t *> PyMem_Malloc(nnz * sizeof(INT64_t))
        if not col:
            PyMem_Free(val)
            raise MemoryError()

        link = <INT64_t *> PyMem_Malloc(nnz * sizeof(INT64_t))
        if not link:
            PyMem_Free(val)
            PyMem_Free(col)
            raise MemoryError()

        root = <INT64_t *> PyMem_Malloc(nrow * sizeof(INT64_t))
        if not root:
            PyMem_Free(val)
            PyMem_Free(col)
            PyMem_Free(link)
            raise MemoryError()

        end_root = <INT64_t *> PyMem_Malloc(nrow * sizeof(INT64_t))
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
        nnz_real = 0
        for nnz_read from 0 <= nnz_read < nnz:
            #print "read element... "
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


            v = <INT64_t>atoi(token_list[2])


            if use_zero_storage or v != 0.0:
                nnz_real = nnz_real + 1
                #############################
                # fill in arrays
                #############################

                if root[i] == -1:
                    #print "first element on the row %d" % i
                    # first element on row i
                    root[i] = nnz_read

                col[nnz_read] = j
                val[nnz_read] = v
                # last element on row i
                end_root[i] = nnz_read
                link[end_root[i]] = nnz_read + 1

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
        A.__nnz = nnz_real
        A.nalloc = nnz


        if nnz != nnz_read:
            raise IOError('Matrix Market file contains %d data lines instead of %d' % (nnz_read, nnz))

    return A
