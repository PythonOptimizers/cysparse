.. io_format:

==========================================================
IO formats
==========================================================

Several IO formats are supported.

Text formats
==================================

Matrix Market format
---------------------

The Matrix Market format is detailed in the `Matrix Market <http://math.nist.gov/MatrixMarket/index.html>`_ website.

The different types of matrices that are supported are basically declared on the first line of the text file in what is called the *Matrix Market banner*: 

    %%MatrixMarket matrix SECOND_FIELD THIRD_FIELD FOURTH_FIELD
    
The start of the banner ``%%MatrixMarket matrix`` is mandatory. We detail the other fields:

- ``SECOND_FIELD``: ``coordinate`` for sparse matrices and ``array`` for dense matrices;
- ``THIRD_FIELD``: ``complex`` or ``real`` for the type of elements. Elements as stored as ``C-double``. ``integer`` denotes integers and ``pattern`` is used 
  when only the pattern of the matrix is given, i.e. only non zero values indices are given and not values. 
- ``FOURTH_FIELD``: denotes the storage scheme: ``general``, ``hermitian``, ``symmetric`` or ``skew``.


..  warning:: Matrix Market matrices are **always** 1-based, i.e. the index of the first element of a matrix is ``(1, 1)`` not ``(0, 0)``.
