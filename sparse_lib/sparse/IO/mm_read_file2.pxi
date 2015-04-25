from sparse_lib.sparse.ll_mat cimport LLSparseMatrix

from libc.stdio cimport *


cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile(char * filename):
    """

    :param filename:
    :return:
    """
    cdef FILE *p
    p = fopen(filename, "r")
    if p == NULL:
        raise IOError("Couldn't open the Matrix Market file '%s'" % filename)

    cdef int nrow, ncol

    status = mm_read_mtx_array_size(p, &nrow, &ncol)
    if status != 0:
        raise IOError("Could not process Matrix Market banner in file '%s'" % filename)

    print "nrow = %d, ncol = %d" % (nrow, ncol)
    fclose(p)

########################################################################################################################
# Matrix Market I/O library from http://math.nist.gov/MatrixMarket
########################################################################################################################
cdef mm_clear_typecode(MM_typecode *typecode):
    (typecode[0])[0]=(typecode[0])[1]= (typecode[0])[2]=' '
    (typecode[0])[3]='G'

cdef int mm_read_banner(FILE *f, MM_typecode *matcode):
    cdef:
        char line[MM_MAX_LINE_LENGTH];
        char banner[MM_MAX_TOKEN_LENGTH];
        char mtx[MM_MAX_TOKEN_LENGTH];
        char crd[MM_MAX_TOKEN_LENGTH];
        char data_type[MM_MAX_TOKEN_LENGTH];
        char storage_scheme[MM_MAX_TOKEN_LENGTH];
        char *p;


    mm_clear_typecode(matcode)

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL):
        return MM_PREMATURE_EOF

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5):
        return MM_PREMATURE_EOF

    # convert to lower case
    p = mtx
    i = 0
    while (p != '\0'):p
        p[i] = tolower(p[i])

        i += 1
    #for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
    # for (p=crd; *p!='\0'; *p=tolower(*p),p++);
    # for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
    # for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);
    #
    # /* check for banner */
    # if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
    #     return MM_NO_HEADER;
    #
    # /* first field should be "mtx" */
    # if (strcmp(mtx, MM_MTX_STR) != 0)
    #     return  MM_UNSUPPORTED_TYPE;
    # mm_set_matrix(matcode);
    #
    #
    # /* second field describes whether this is a sparse matrix (in coordinate
    #         storgae) or a dense array */
    #
    #
    # if (strcmp(crd, MM_SPARSE_STR) == 0)
    #     mm_set_sparse(matcode);
    # else
    # if (strcmp(crd, MM_DENSE_STR) == 0)
    #         mm_set_dense(matcode);
    # else
    #     return MM_UNSUPPORTED_TYPE;
    #
    #
    # /* third field */
    #
    # if (strcmp(data_type, MM_REAL_STR) == 0)
    #     mm_set_real(matcode);
    # else
    # if (strcmp(data_type, MM_COMPLEX_STR) == 0)
    #     mm_set_complex(matcode);
    # else
    # if (strcmp(data_type, MM_PATTERN_STR) == 0)
    #     mm_set_pattern(matcode);
    # else
    # if (strcmp(data_type, MM_INT_STR) == 0)
    #     mm_set_integer(matcode);
    # else
    #     return MM_UNSUPPORTED_TYPE;
    #
    #
    # /* fourth field */
    #
    # if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
    #     mm_set_general(matcode);
    # else
    # if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
    #     mm_set_symmetric(matcode);
    # else
    # if (strcmp(storage_scheme, MM_HERM_STR) == 0)
    #     mm_set_hermitian(matcode);
    # else
    # if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
    #     mm_set_skew(matcode);
    # else
    #     return MM_UNSUPPORTED_TYPE;


    return 0





cdef int mm_read_mtx_array_size(FILE *f, int *M, int *N):
    cdef:
        char line[MM_MAX_LINE_LENGTH];
        int num_items_read;

    # set return null parameter values, in case we exit with errors
    M[0] = N[0] = 0

    # now continue scanning until you reach the end-of-comments
    if fgets(line,MM_MAX_LINE_LENGTH,f) == NULL:
        return MM_PREMATURE_EOF

    while line[0] == '%':
        if fgets(line,MM_MAX_LINE_LENGTH,f) == NULL:
            return MM_PREMATURE_EOF

    # line[] is either blank or has M,N, nz
    if sscanf(line, "%d %d", M, N) == 2:
        return 0

    else: # we have a blank line
        num_items_read = fscanf(f, "%d %d", M, N)
        if num_items_read == EOF:
            return MM_PREMATURE_EOF

    while num_items_read != 2:
        num_items_read = fscanf(f, "%d %d", M, N)
        if num_items_read == EOF:
            return MM_PREMATURE_EOF

    return 0
