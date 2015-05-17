from cysparse.sparse.ll_mat cimport LLSparseMatrix

from libc.stdio cimport *

import numpy as np
from string import atoi, atof


class MatrixMarketMatrix:
    """
    A MatrixMarketMatrix object represents a sparse matrix read from a file.
    This file must describe a sparse matrix in the MatrixMarket file format.

    See http://math.nist.gov/MatrixMarket for more information.

    Example: mat = MatrixMarketMatrix('1138bus.mtx')
    """
    # TODO: add Cython types!!!
    # TODO: generalize to other types of files: .gz
    # TODO: take into account matrix type
    # TODO: write directly into a LLSparseMatrix

    def __init__(self, fname, **kwargs):
        self.comments= ''
        self.dtype = None
        self.irow = None
        self.jcol = None
        self.values = None
        self.nrow = self.ncol = self.nnz = 0
        self.shape = (0,0)
        self.symmetric = self.Hermitian = self.skewsym = False

        fp = open(fname)
        pos = self._readHeader(fp)
        nrecs = self._readData(fp,pos)
        fp.close()

        if nrecs != self.nnz:
            raise ValueError, 'Read %d records. Expected %d.' % (nrecs,self.nnz)

    def _readHeader(self, fp):
        fp.seek(0)
        hdr = fp.readline().split()
        if hdr[1] != 'matrix' or hdr[2] != 'coordinate':
            raise TypeError, 'Type not supported: %s' % hdr[1:3]

        # Determine entries type
        dtypes = {'real': np.float,
                  'complex': np.complex,
                  'integer': np.int,
                  'pattern': None}

        self.dtype = dtypes[hdr[3]]

        # Determine symmetry
        if hdr[4] == 'symmetric':
            self.symmetric = True
        elif hdr[4] == 'Hermitian':
            self.Hermitian = True
        elif hdr[4] == 'skew-symmetric':
            self.skewsym = True

        # Read comments
        line = fp.readline()
        while line[0] == '%':
            self.comments += line
            line = fp.readline()

        # Return current position
        return fp.tell() - len(line)

    def _readData(self, fp, pos):
        fp.seek(pos)
        size = fp.readline().split()
        self.nrow = atoi(size[0])
        self.ncol = atoi(size[1])
        self.shape = (self.nrow, self.ncol)
        self.nnz  = atoi(size[2])
        self.irow = np.empty(self.nnz, dtype=np.int)
        self.jcol = np.empty(self.nnz, dtype=np.int)

        if self.dtype is not None:
            self.values = np.empty(self.nnz, dtype=self.dtype)

        # Read in data
        k = 0
        for line in fp.readlines():
            line = line.split()
            self.irow[k] = atoi(line[0])-1
            self.jcol[k] = atoi(line[1])-1
            if self.dtype == np.int:
                self.values[k] = atoi(line[2])
            elif self.dtype == np.float:
                self.values[k] = atof(line[2])
            elif self.dtype == np.complex:
                self.values[k] = complex(atof(line[2]),atof(line[3]))
            k += 1

        return k

    def find(self):
        """
        Return the sparse matrix in triple format (val,irow,jcol). If the
        matrix data type is `None`, i.e., only the matrix sparsity pattern
        is available, this method returns (irow,jcol).
        """
        if self.dtype is not None:
            return (self.values,self.irow,self.jcol)
        return (self.irow,self.jcol)

cdef LLSparseMatrix MakeLLSparseMatrixFromMMFile(str mm_filename):
    """

    :param filename:
    :return:
    """
    # TODO: rewrite function and optimize!
    #cdef FILE *p
    #p = fopen(filename, "r")
    #if p == NULL:
    #    raise IOError("Couldn't open the Matrix Market file '%s'" % filename)

    #cdef int nrow, ncol

    #fclose(p)

    mm = MatrixMarketMatrix(mm_filename)
    result = mm.find()
    if len(result) != 3:
        raise NotImplementedError("Cannot read that type of Matrix Market file")

    cdef LLSparseMatrix A
    A = LLSparseMatrix(nrow=mm.nrow, ncol=mm.ncol, size_hint=mm.nnz)
    for val, i, j in zip(*result):
        A.safe_put(i, j, val)

    return A