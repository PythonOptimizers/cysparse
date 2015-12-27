
..  _matrix_creation:

========================
How to create a matrix?
========================

To create a matrix, we use *factory methods* [#factory_method_strange_name]_: 
functions that return an object corresponding
to their arguments. Different arguments make them return different kind of objects (matrices).

..  warning::  An sparse matrix can **only** be instantiated through a factory method.

Before you can use any type of sparse matrix, you **must** first instantiate an ``LLSparseMatrix``. This matrix is well suited for construction but is not very optimized for most matrix operations. 
Once you have an ``LLSparseMatrix``, you can create a specialized sparse matrix from it.

..  warning:: The first matrix you are allowed to instantiate can **only** be an ``LLSparseMatrix``.

So basically, you create an ``LLSparseMatrix``, populate it by for instance add and/or delete elements, rows, columns, sub-matrices at will. Once you have constructed your matrix, it is time to transform it into an appropriate 
matrix format that is optimized for your needs. This transformation is not done in place and a copy is made. Here is an example:

..  code-block:: python

    A = ...  # A is a LLSparseMatrix
    # add some elements
    for i in range(n):
        for j in range(m):
            A[i, j] = ...
    
    # once the matrix is constructed, transform it into suitable matrix format
    # here to CSC
    C = A.to_csc()

A ``CSCSparseMatrix`` can also be transformed into a ``CSRSparseMatrix`` for instance:

..  code-block:: python

    # transform a CSCSparseMatrix into a CSRSparseMatrix
    D = C.to_csr()

But the first matrix **must** be an ``LLSparseMatrix``.
    
Before we look how to create any matrix like object from an ``LLSparseMatrix`` or other sparse objects, let's look at the different ways to create the first ``LLSparseMatrix`` in the next section.


``LLSparseMatrix`` factory methods
===========================================


``LLSparseMatrix``
----------------------

This is the main basic factory method to create an :class:`LLSparseMatrix`. There are basically **three** ways to use it:

- From specifications. Use ``nrow`` and ``ncol`` or ``size`` to specify the dimension of the new matrix. You can also provide a
  ``size_hint`` argument to (pre)allocate some space for the elements in advance.
  
  ..  code-block:: python

      A = LLSparseMatrix(nrow=256, ncol=3398, size_hint=600)

  which returns an empty ``LLSparseMatrix [INT64_t, FLOAT64_t] of size=(256, 3398) with 0 non zero values <Storage scheme: General and without zeros>`` matrix that is ready to hold 600 elements of type ``FLOAT64_t``.
  Notice that neither ``size`` nor ``size_hint`` are attributes.
  
  You can change the index type and/or the element type:
  
  ..  code-block:: python

      A = LLSparseMatrix(size=5578, size_hint=600, itype=INT32_T, dtype=COMPLEX128_T)
      
  which returns a corresponding ``LLSparseMatrix [INT32_t, COMPLEX128_t] of size=(5578, 5578) with 0 non zero values <Storage scheme: General and without zeros>`` matrix.
  
  The ``itype`` and ``dtype`` arguments represent respectively the index and element types. By default, the index and element types are (``INT32_t``, ``FLOAT32_T``) and (``INT64_t``, ``FLOAT64_T``) on 32bits and 64bits 
  platforms [#modify_default_types_in_cysparse_cfg]_. 
  
- From another matrix. Use the ``matrix`` argument. 

  ..  warning:: This is not yet implemented.

- From a file. By default, a ``test_bounds`` argument is set to ``True`` to test if all indices are not out of bounds. You can disable this to gain
  some speed when reading a file.
  For the moment, only the `Matrix Market <http://math.nist.gov/MatrixMarket/>`_ format is available. To give a file name of a file in Matrix Market format, use the ``mm_filename`` argument.           
  
  ..  code-block:: python
  
      A = LLSparseMatrix(mm_filename='bcsstk01.mtx', itype=INT64_T, dtype=FLOAT64_T)
      
  This will create a ``LLSparseMatrix [INT64_t, FLOAT64_t] of size=(48, 48) with 224 non zero values <Storage scheme: Symmetric and without zeros>`` matrix.
      
  To read a matrix file, you need to already know the type of your ``LLSparseMatrix``, here we choose ``itype=INT64_T, dtype=FLOAT64_T``. If you don't know the exact type of the matrix you need to read your file, use :func:`LLSparseMatrixFromMMFile` instead. This
  factory method will return the right ``LLSparseMatrix`` for you. 
  
``LLSparseMatrixFromMMFile``
-----------------------------

We just saw the ``LLSpasreMatrix`` factory method. This factory method can only be used if you know by advance the **exact** type of your matrix object. It might happen that you don't know exactly what type of matrix you need. 
This is especially true when you read a file to populate a matrix. ``LLSparseMatrixFromMMFile`` allows you to open any Matrix Market file and create the right type of ``LLSparseMatrix``, i.e. the minimal ``LLSparseMatrix`` type
to hold the matrix. Let try with the same file :file:`bcsstk01.mtx` as above:

..  code-block:: python

    A = LLSparseMatrixFromMMFile('bcsstk01.mtx')
    
This returns a ``LLSparseMatrix [INT32_t, FLOAT64_t] of size=(48, 48) with 224 non zero values <Storage scheme: Symmetric and without zeros>`` matrix. Indeed, we didn't need to use ``INT64_t`` for the index type, ``INT32_t`` is 
sufficient.

This factory method also accepts two more arguments:

- ``store_zero``: ``False`` by default and
- ``test_bounds``: ``True`` by default.

Let try them:

..  code-block:: python

    A = LLSparseMatrixFromMMFile('bcsstk01.mtx', store_zero=True, test_bounds=False)
    
``A`` is now an ``LLSparseMatrix [INT32_t, FLOAT64_t] of size=(48, 48) with 224 non zero values <Storage scheme: Symmetric and with zeros>``. We now store explicitly zeros. We didn't test if the elements where 
all within a :math:`48 \times 48` matrix but we already knew that this is the case, so we could speed up the reading by setting ``test_bounds`` to ``False``.    

``DiagonalLLSparseMatrix``
------------------------------

``DiagonalLLSparseMatrix`` constructs `diagonal matrix <https://en.wikipedia.org/wiki/Diagonal_matrix>`_.
You can give the diagonal element with the ``element`` argument:

..  code-block:: python

    A = DiagonalLLSparseMatrix(element=3-5j, nrow=2, ncol=3, dtype=COMPLEX128_T)

This returns:

..  code-block:: python

    LLSparseMatrix [INT64_t, COMPLEX128_t] of size=(2, 3) with 2 non zero values 
    <Storage scheme: General and without zeros>
     3.000000 - 5.000000j    ---        ---        ---        ---     
       ---        ---      3.000000 - 5.000000j    ---        ---   
       
If no element is given, a default of ``1`` is used.

``IdentityLLSparseMatrix``
------------------------------

This factory method creates the special diagonal matrix with only ones on its main diagonal. The `identity matrix <https://en.wikipedia.org/wiki/Identity_matrix>`_ 

For instance:

..  code-block:: python

    A = IdentityLLSparseMatrix(size = 3, dtype=COMPLEX64_T)
    print A

returns:

..  code-block:: python

    LLSparseMatrix [INT64_t, COMPLEX64_t] of size=(3, 3) with 3 non zero values 
    <Storage scheme: General and without zeros>
     1.000000 + 0.000000j    ---        ---        ---        ---     
       ---        ---      1.000000 + 0.000000j    ---        ---     
       ---        ---        ---        ---      1.000000 + 0.000000j     

Note that this factory method is only an alias for ``DiagonalLLSparseMatrix`` with default arguments.

``BandLLSparseMatrix``
---------------------------

``BandLLSparseMatrix`` creates `band matrices <https://en.wikipedia.org/wiki/Band_matrix>`_ but not only as you can use it to create more general "band-like" matrices. In :program:`CySparse`, you can use this factory 
method to create matrices with principal diagonals. To do so, you create some :program:`NumPy` vectors and you assign them to the diagonals. Here is an example:

..  code-block:: python

    b1 = np.array([1+1j], dtype=np.complex64)
    b2 = np.array([1+1j, 12-1j], dtype=np.complex64)
    b3 = np.array([1+1j, 1+1j, -1j], dtype=np.complex64)   

    A = BandLLSparseMatrix(nrow=2, ncol=3, diag_coeff=[-1, 1, 2], numpy_arrays=[b1, b2, b3], dtype=COMPLEX64_T)
    print A 

wich prints:

..  code-block:: python

    LLSparseMatrix [INT64_t, COMPLEX64_t] of size=(2, 3) with 4 non zero values 
    <Storage scheme: General and without zeros>
       ---        ---      1.000000 + 1.000000j  1.000000 + 1.000000j 
     1.000000 + 1.000000j    ---        ---     12.000000 - 1.000000j 

As you can see, we didn't create a real band matrix. The diagonals to fill in are given in list in the ``diag_coeff`` argument and the :program:`NumPy` vectors are given inside another list in the ``numpy_arrays``
argument. The diagonals can also be given in a slice.

You also can notice that the :program:`NumPy` vectors can "overflow": :program:`CySparse` only take the first elements of a given vector to populate a given diagonal. 
These vectors **must** be big enough to fill the diagonals.
 
``ArrowheadLLSparseMatrix``
-------------------------------

``LinearFillLLSparseMatrix``
------------------------------

``PermutationLLSparseMatrix``
-------------------------------

Other ways to create matrices or matrix like objects
=======================================================



..  raw:: html

    <h4>Footnotes</h4>

..  [#factory_method_strange_name] The term *factory method* is coined by the Design Pattern community. The *method* in itself can be a function, method, class, ... In :program:`CySparse`, we use global functions.
    This not very Pythonesque approach is made necessary because :program:`Cython` doesn't allow the use of pure C variables as arguments in the constructors of classes [#use_of_pure_c_variables_in_constructors]_.
    
..  [#use_of_pure_c_variables_in_constructors] This not exactly true. :program:`Cython` allows to pass some pure C variables that can be *easily* mapped to :program:`Python` arguments. The idea is that the same arguments are 
    passed to ``__cinit__()`` **and** ``__init__()`` methods. 
    
..  [#modify_default_types_in_cysparse_cfg] You can change this default behavior by giving other values in the :file:`cysparse.cfg` configuration file. See :ref:`cysparse_configuration_file`.   

