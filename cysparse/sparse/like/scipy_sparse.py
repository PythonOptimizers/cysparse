"""
Try to emulate the scipy.sparse library.

This module is subject to frequent changes.

THIS IS WORK IN PROGRESS
"""
# TODO: this doesn't work... why?
import numpy as np


def diags(diagonals, offsets, shape=None, format=None, dtype=None):
    """
    Construct a sparse matrix from diagonals.
    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the matrix diagonals,
        corresponding to `offsets`.
    offsets : sequence of int
        Diagonals to set:
          - k = 0  the main diagonal
          - k > 0  the k-th upper diagonal
          - k < 0  the k-th lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result.  By default (format=None) an
        appropriate sparse matrix format is returned.  This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the matrix.

    Examples
    --------
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags(diagonals, [0, -1, 2])
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])
    Broadcasting of scalars is supported (but shape needs to be
    specified):
    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])
    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:
    >>> diags([1, 2, 3], 1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    pass